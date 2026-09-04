// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/poolwire"
)

// tunnelDialTimeout bounds connecting to the local backend on the donor side and
// to the donor on the head side. Short on the loopback leg because a process on
// this machine either accepts immediately or is not there; longer on the network
// leg is unnecessary because the peer's address was already confirmed reachable
// before any of this was attempted.
const tunnelDialTimeout = 10 * time.Second

// tunnelHandshakeTimeout bounds the HTTP upgrade exchange. Once the stream is
// established there are NO deadlines on it: a ggml transfer can be idle for as
// long as the head takes to prepare the next batch, and a deadline would sever
// a working pool for being quiet.
const tunnelHandshakeTimeout = 30 * time.Second

// halfCloser is the subset of a connection that can signal end-of-stream in one
// direction. Both *net.TCPConn and *tls.Conn implement it.
//
// Half-close is what makes the splice below correct rather than merely usable.
// When one side finishes sending, the other must see EOF; closing the whole
// connection instead would destroy the reply that is still in flight, which for
// a tensor transfer means a truncated buffer rather than a clean end.
type halfCloser interface{ CloseWrite() error }

// closeWrite signals end-of-stream in one direction where the connection
// supports it, and does nothing where it does not.
func closeWrite(c net.Conn) {
	if hc, ok := c.(halfCloser); ok {
		_ = hc.CloseWrite()
	}
}

// splice joins two connections into one bidirectional stream and blocks until
// both directions have ended.
//
// buffered is whatever a bufio.Reader already pulled off `a` while reading the
// upgrade — on a fast client the first tensor bytes can arrive in the same
// segment as the request, and reading `a` directly afterwards would silently
// drop them. Reading through the bufio.Reader for the whole life of the stream
// is what avoids that; it is not an optimization.
func splice(a net.Conn, buffered *bufio.Reader, b net.Conn) {
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		_, _ = io.Copy(b, buffered)
		closeWrite(b)
	}()
	go func() {
		defer wg.Done()
		_, _ = io.Copy(a, b)
		closeWrite(a)
	}()
	wg.Wait()
}

// handleTunnel upgrades a peer's request into the ggml RPC stream for one lease.
//
// This is the single most security-sensitive handler in PAIR, and it is worth
// being explicit about why. On the far side of the splice is `rpc-server`, a
// protocol with no authentication that deserializes tensor and buffer
// descriptors straight into a compute backend; upstream llama.cpp is unambiguous
// that it must never be exposed to an untrusted network. Everything that makes
// that safe here happens before the splice: the connection is cluster mTLS, the
// caller presents a certificate this node currently pins, the lease exists, and
// the caller is the node the lease was issued to. The backend itself binds
// loopback and is unreachable any other way.
func (s *Server) handleTunnel(w http.ResponseWriter, r *http.Request) {
	holder, ok := s.authorize(w, r)
	if !ok {
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	leaseID, ok := pathSuffix(r.URL.Path, poolwire.TunnelPathPrefix)
	if !ok {
		http.Error(w, "bad lease id", http.StatusBadRequest)
		return
	}
	// Require the protocol token. A request that reaches here without it is not
	// a pool tunnel — most likely something misrouted — and splicing it into a
	// tensor backend on the strength of the path alone would be the wrong kind
	// of forgiving.
	if !headerHasToken(r.Header.Values("Upgrade"), poolwire.TunnelProtocol) {
		http.Error(w, "expected an "+poolwire.TunnelProtocol+" upgrade", http.StatusBadRequest)
		return
	}
	if s.leases == nil {
		http.Error(w, "pooling is not available on this node", http.StatusServiceUnavailable)
		return
	}

	target, err := s.leases.authorizeTunnel(holder, leaseID)
	if err != nil {
		// Deliberately the same answer for "no such lease" and "not yours": the
		// distinction would confirm the existence of a lease the caller has no
		// business knowing about.
		http.Error(w, "no such lease", http.StatusNotFound)
		return
	}

	// Dial the backend BEFORE hijacking, so a backend that is not there is
	// reported as an HTTP status the head can act on rather than as a
	// connection that opens and immediately dies.
	upstream, err := net.DialTimeout("tcp", target, tunnelDialTimeout)
	if err != nil {
		slog.Error("pool tunnel: local backend unreachable", "lease", leaseID, "target", target, "err", err)
		http.Error(w, "donor backend unavailable", http.StatusBadGateway)
		return
	}

	hijacker, ok := w.(http.Hijacker)
	if !ok {
		_ = upstream.Close()
		http.Error(w, "connection cannot be upgraded", http.StatusInternalServerError)
		return
	}
	conn, brw, err := hijacker.Hijack()
	if err != nil {
		_ = upstream.Close()
		slog.Error("pool tunnel: hijack failed", "lease", leaseID, "err", err)
		return
	}

	// The listener's read and idle timeouts covered the handshake. From here the
	// stream is long-lived and may be legitimately idle, so every deadline is
	// cleared; teardown comes from the lease, not from a clock.
	_ = conn.SetDeadline(time.Time{})

	untrack, live := s.leases.trackConn(leaseID, func() {
		_ = conn.Close()
		_ = upstream.Close()
	})
	if !live {
		// The lease was released between authorization and here.
		_ = conn.Close()
		_ = upstream.Close()
		return
	}
	defer untrack()

	if _, err := brw.WriteString("HTTP/1.1 101 Switching Protocols\r\n" +
		"Upgrade: " + poolwire.TunnelProtocol + "\r\n" +
		"Connection: Upgrade\r\n\r\n"); err != nil {
		_ = conn.Close()
		_ = upstream.Close()
		return
	}
	if err := brw.Flush(); err != nil {
		_ = conn.Close()
		_ = upstream.Close()
		return
	}

	slog.Info("pool tunnel open", "lease", leaseID, "holder", holder, "target", target)
	splice(conn, brw.Reader, upstream)
	_ = conn.Close()
	_ = upstream.Close()
	slog.Info("pool tunnel closed", "lease", leaseID)
}

// DonorLink is the head side of a tunnel: a loopback listener that llama.cpp is
// pointed at, carrying every connection to one donor over cluster mTLS.
//
// It exists because llama.cpp cannot speak TLS. `--rpc host:port` opens a plain
// TCP connection and sends the ggml protocol on it, so the only way to keep that
// protocol off the network is to give it a local address that is not on the
// network — and then to be the thing that crosses the network on its behalf.
type DonorLink struct {
	mesh     *clustertrust.Mesh
	peerUUID string
	// peerAddr is the donor's cluster port, already confirmed reachable by the
	// caller. A DonorLink does not walk candidates: by the time a pool is
	// forming, capacity has been collected from this peer at a known-good
	// address, and re-deriving it here would pick a different one.
	peerAddr string
	leaseID  string

	ln     net.Listener
	wg     sync.WaitGroup
	closed chan struct{}
	once   sync.Once
}

// NewDonorLink prepares a link. Nothing is bound until Start.
func NewDonorLink(mesh *clustertrust.Mesh, peerUUID, peerAddr, leaseID string) *DonorLink {
	return &DonorLink{
		mesh:     mesh,
		peerUUID: peerUUID,
		peerAddr: peerAddr,
		leaseID:  leaseID,
		closed:   make(chan struct{}),
	}
}

// Start binds the loopback listener and begins accepting.
//
// 127.0.0.1 explicitly, never a wildcard bind. The whole point of this listener
// is to be an address only this machine can reach, so that the plaintext ggml
// protocol has nowhere to leak to; binding it on all interfaces would recreate
// exactly the exposure the tunnel exists to remove.
func (d *DonorLink) Start(ctx context.Context) error {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return fmt.Errorf("bind donor link: %w", err)
	}
	d.ln = ln

	d.wg.Add(1)
	go func() {
		defer d.wg.Done()
		for {
			conn, err := ln.Accept()
			if err != nil {
				select {
				case <-d.closed:
				default:
					slog.Debug("donor link accept ended", "lease", d.leaseID, "err", err)
				}
				return
			}
			d.wg.Add(1)
			go func() {
				defer d.wg.Done()
				d.carry(ctx, conn)
			}()
		}
	}()
	slog.Info("donor link listening", "lease", d.leaseID, "addr", ln.Addr().String(), "peer", d.peerUUID)
	return nil
}

// Addr is the loopback address to hand llama.cpp as an --rpc endpoint.
func (d *DonorLink) Addr() string {
	if d.ln == nil {
		return ""
	}
	return d.ln.Addr().String()
}

// Close stops accepting and tears down every carried connection.
func (d *DonorLink) Close() error {
	d.once.Do(func() { close(d.closed) })
	if d.ln != nil {
		_ = d.ln.Close()
	}
	d.wg.Wait()
	return nil
}

// carry moves one local connection to the donor over cluster mTLS.
func (d *DonorLink) carry(ctx context.Context, local net.Conn) {
	defer func() { _ = local.Close() }()

	remote, buffered, err := d.dialTunnel(ctx)
	if err != nil {
		slog.Error("donor link could not reach the donor", "lease", d.leaseID, "peer", d.peerUUID, "err", err)
		return
	}
	defer func() { _ = remote.Close() }()

	// Closing the link must sever carried streams too; otherwise a torn-down
	// pool leaves a tensor backend attached to a head that has moved on.
	done := make(chan struct{})
	go func() {
		select {
		case <-d.closed:
			_ = local.Close()
			_ = remote.Close()
		case <-done:
		}
	}()
	defer close(done)

	splice(remote, buffered, local)
}

// dialTunnel opens the mTLS connection and performs the upgrade, returning the
// connection and the buffered reader that must be used for every subsequent read
// from it.
func (d *DonorLink) dialTunnel(ctx context.Context) (net.Conn, *bufio.Reader, error) {
	// Resolve the client config per dial rather than once at construction: a
	// re-pairing mints a new leaf, and a config built around the old one keeps
	// presenting an identity the donor has stopped accepting.
	d.mesh.Refresh()
	cfg, ok := d.mesh.ClientTLSConfig(d.peerUUID)
	if !ok {
		return nil, nil, fmt.Errorf("no pin for donor %s", d.peerUUID)
	}

	dialer := &tls.Dialer{NetDialer: &net.Dialer{Timeout: tunnelDialTimeout}, Config: cfg}
	dialCtx, cancel := context.WithTimeout(ctx, tunnelHandshakeTimeout)
	defer cancel()

	conn, err := dialer.DialContext(dialCtx, "tcp", d.peerAddr)
	if err != nil {
		return nil, nil, fmt.Errorf("dial donor: %w", err)
	}

	url := "https://" + d.peerAddr + poolwire.TunnelPathPrefix + d.leaseID
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, nil)
	if err != nil {
		_ = conn.Close()
		return nil, nil, fmt.Errorf("build upgrade: %w", err)
	}
	req.Header.Set("Connection", "Upgrade")
	req.Header.Set("Upgrade", poolwire.TunnelProtocol)

	// The upgrade is written and read by hand rather than through http.Client,
	// because a Client owns its connection and never gives it back. A tunnel
	// needs the raw stream after the response, which is the one thing the
	// high-level API cannot provide.
	_ = conn.SetDeadline(time.Now().Add(tunnelHandshakeTimeout))
	if err := req.Write(conn); err != nil {
		_ = conn.Close()
		return nil, nil, fmt.Errorf("send upgrade: %w", err)
	}
	buffered := bufio.NewReader(conn)
	resp, err := http.ReadResponse(buffered, req)
	if err != nil {
		_ = conn.Close()
		return nil, nil, fmt.Errorf("read upgrade response: %w", err)
	}
	if resp.StatusCode != http.StatusSwitchingProtocols {
		_ = resp.Body.Close()
		_ = conn.Close()
		return nil, nil, fmt.Errorf("donor refused the tunnel: %s", resp.Status)
	}
	// Cleared for the same reason as on the donor side: a live pool may be idle.
	_ = conn.SetDeadline(time.Time{})
	return conn, buffered, nil
}

// pathSuffix returns the single path segment after prefix.
//
// It refuses anything containing a separator rather than taking the first
// segment. The value becomes a lookup key for a capability, and a lenient parse
// is how "/v1/pool/rpc/abc/../def" turns into a lease somebody else holds.
func pathSuffix(path, prefix string) (string, bool) {
	if len(path) <= len(prefix) || path[:len(prefix)] != prefix {
		return "", false
	}
	rest := path[len(prefix):]
	if rest == "" || strings.Contains(rest, "/") {
		return "", false
	}
	return rest, true
}

// headerHasToken reports whether any of values carries token, case-insensitively,
// allowing the comma-separated list form headers permit.
func headerHasToken(values []string, token string) bool {
	for _, v := range values {
		for _, part := range strings.Split(v, ",") {
			if strings.EqualFold(strings.TrimSpace(part), token) {
				return true
			}
		}
	}
	return false
}
