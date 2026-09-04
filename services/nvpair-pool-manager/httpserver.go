// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/poolwire"
)

// readHeaderTimeout bounds how long a peer may take to send its request
// headers.
const readHeaderTimeout = 5 * time.Second

// shutdownGrace bounds the drain on shutdown.
const shutdownGrace = 3 * time.Second

// Server is the cluster-scoped HTTP surface: what this node will contribute to
// a pool, and later the donor leases and their tunnels.
//
// There is no plaintext personality, unlike the proxies and cluster-manager. The
// two reasons compound. Everything served here is cluster state — what a user's
// machines hold and what they have promised each other — which nvpair-node-info's
// documented plaintext trade explicitly does not cover. And the same listener
// carries the ggml tunnel, a protocol with no authentication of its own whose
// entire safety rests on nothing unpinned being able to open it. A port that is
// mTLS-only in every state cannot be misconfigured into being neither.
type Server struct {
	port      int
	mesh      *clustertrust.Mesh
	collector *Collector
	// leases is the donor side. Nil on a node built without pooling support, in
	// which case the lease and tunnel endpoints answer 503 rather than being
	// absent — a head asking a donor that cannot serve deserves a reason, not a
	// 404 it will read as a bad address.
	leases *LeaseStore

	srv *http.Server

	// addr is the address actually bound, published before bound is closed so a
	// reader that has seen the close can read it without synchronization. It
	// exists because the port may be 0 — the broker passes a fixed port, but a
	// test that hard-coded one could not run twice concurrently, and a service
	// whose only listener cannot be exercised end to end tends to grow handlers
	// nobody has dialed.
	addr  string
	bound chan struct{}
}

func NewServer(port int, mesh *clustertrust.Mesh, collector *Collector, leases *LeaseStore) *Server {
	return &Server{port: port, mesh: mesh, collector: collector, leases: leases, bound: make(chan struct{})}
}

// Bound is closed once the listener has a port. Addr is readable from then on.
func (s *Server) Bound() <-chan struct{} { return s.bound }

// Addr reports the bound address, or "" before Bound is closed.
func (s *Server) Addr() string {
	select {
	case <-s.bound:
		return s.addr
	default:
		return ""
	}
}

// Run binds the port once and serves cluster mTLS on it for the life of the
// process.
//
// The server certificate is resolved per handshake from the live mesh, which is
// what lets one listener cover both membership states: while unclustered it
// presents no leaf and every handshake is refused, and the moment this node
// joins, the same listener serves pinned peers. No rebind and no restart to
// re-read the cluster directory — which matters because a user can enable
// donation before they have paired anything, and the port must start working
// when they do.
func (s *Server) Run(ctx context.Context) error {
	mux := http.NewServeMux()
	mux.HandleFunc(poolwire.CapacityPath, s.handleCapacity)
	mux.HandleFunc(poolwire.LeasesPath, s.handleLeases)
	mux.HandleFunc(poolwire.LeasePathPrefix, s.handleLease)
	mux.HandleFunc(poolwire.TunnelPathPrefix, s.handleTunnel)

	base, err := net.Listen("tcp", ":"+strconv.Itoa(s.port))
	if err != nil {
		return fmt.Errorf("listen on :%d: %w", s.port, err)
	}
	s.addr = base.Addr().String()
	close(s.bound)

	s.srv = &http.Server{
		Handler:           mux,
		ReadHeaderTimeout: readHeaderTimeout,
		IdleTimeout:       clustertrust.PeerListenerIdleTimeout,
	}

	errCh := make(chan error, 1)
	go func() {
		slog.Info("pool server listening (cluster mTLS)",
			"addr", base.Addr().String(), "clustered", s.mesh.Clustered())
		if serveErr := s.srv.Serve(tls.NewListener(base, s.mesh.ServerTLSConfig())); serveErr != nil && serveErr != http.ErrServerClosed {
			errCh <- serveErr
		}
	}()

	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownGrace)
		defer cancel()
		_ = s.srv.Shutdown(shutdownCtx)
		return nil
	case err := <-errCh:
		return err
	}
}

// handleCapacity answers a peer's question about what this node can contribute.
func (s *Server) handleCapacity(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.authorize(w, r); !ok {
		return
	}
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	capacity := s.collector.Local(r.Context())
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(capacity); err != nil {
		slog.Debug("pool: writing capacity response failed", "err", err)
	}
}

// authorize is the cluster gate every handler on this listener runs first.
//
// It refreshes the mesh before deciding, so a peer paired since the last request
// is recognized without a restart, and a peer removed since the last request
// stops being recognized immediately — which for this service is the difference
// between revoking donation and merely intending to.
//
// It is unconditional. VerifyClientPin is already false for a node that belongs
// to no cluster, so an unclustered node refuses everything here rather than
// falling back to serving it in the clear.
func (s *Server) authorize(w http.ResponseWriter, r *http.Request) (string, bool) {
	s.mesh.Refresh()
	peerUUID, ok := s.mesh.VerifyClientPin(r)
	if !ok {
		http.Error(w, "forbidden: not a pinned cluster peer", http.StatusForbidden)
		return "", false
	}
	return peerUUID, true
}

// leaseMaxBytes caps a lease request body. The document is a few hundred bytes;
// the cap is here because the body arrives from the network.
const leaseMaxBytes = 64 << 10

// handleLeases grants a lease. POST only: a lease is created, never listed —
// a peer has no business enumerating what this node has promised to others.
func (s *Server) handleLeases(w http.ResponseWriter, r *http.Request) {
	holder, ok := s.authorize(w, r)
	if !ok {
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if s.leases == nil {
		http.Error(w, "pooling is not available on this node", http.StatusServiceUnavailable)
		return
	}

	var req poolwire.LeaseRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, leaseMaxBytes)).Decode(&req); err != nil {
		http.Error(w, "invalid lease request", http.StatusBadRequest)
		return
	}

	// Read capacity through the collector, so the answer already accounts for
	// the owner's reservation and for every lease this node has outstanding —
	// including, on a renewal, the one being renewed. Deriving free memory here
	// would be a second implementation of a rule that has to hold everywhere.
	capacity := s.collector.Local(r.Context())
	if !capacity.DonorEnabled {
		http.Error(w, "this node is not lending its GPUs", http.StatusForbidden)
		return
	}
	available := make(map[int]uint64, len(capacity.Devices))
	for _, d := range capacity.Devices {
		available[d.Index] = d.FreeBytes()
	}

	grant, err := s.leases.Grant(r.Context(), holder, req, available)
	if err != nil {
		// A refusal is a normal answer with a reason a user can act on — "device
		// 0 has 6 GiB free, needs 14" is the whole point of asking.
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(grant); err != nil {
		slog.Debug("pool: writing lease grant failed", "err", err)
	}
}

// handleLease renews (POST) or releases (DELETE) one lease.
func (s *Server) handleLease(w http.ResponseWriter, r *http.Request) {
	holder, ok := s.authorize(w, r)
	if !ok {
		return
	}
	if s.leases == nil {
		http.Error(w, "pooling is not available on this node", http.StatusServiceUnavailable)
		return
	}
	leaseID, ok := pathSuffix(r.URL.Path, poolwire.LeasePathPrefix)
	if !ok {
		http.Error(w, "bad lease id", http.StatusBadRequest)
		return
	}

	switch r.Method {
	case http.MethodPost:
		var req poolwire.LeaseRequest
		// A renewal may carry a TTL or nothing at all; an unreadable body takes
		// the default rather than failing, because the body is not what
		// identifies the lease.
		_ = json.NewDecoder(io.LimitReader(r.Body, leaseMaxBytes)).Decode(&req)
		grant, err := s.leases.Renew(holder, leaseID, req.TTL())
		if err != nil {
			http.Error(w, "no such lease", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(grant); err != nil {
			slog.Debug("pool: writing lease renewal failed", "err", err)
		}

	case http.MethodDelete:
		if err := s.leases.Release(holder, leaseID); err != nil {
			http.Error(w, "no such lease", http.StatusNotFound)
			return
		}
		w.WriteHeader(http.StatusNoContent)

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}
