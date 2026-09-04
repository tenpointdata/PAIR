// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"nvpair-shared/poolwire"
)

// LoopbackTarget is the local process a lease's tunnel is spliced to — in
// production the ggml `rpc-server` this node runs for that lease.
//
// It is an interface for one reason that matters more than testability: the
// contract it expresses is that the target is addressed by a LOOPBACK address
// and nothing else. Start returns an address the tunnel dials, and the only
// implementation that may ever be wired in is one that binds 127.0.0.1. The
// ggml RPC protocol has no authentication and deserializes tensor descriptors
// straight into a backend, so a target reachable on any other interface would
// hand an unpinned stranger a memory-unsafe surface — which is precisely the
// mistake the whole tunnel exists to prevent.
type LoopbackTarget interface {
	// Start brings up the local process for a lease and returns its loopback
	// address. It is called once per lease, before the head is told the grant
	// succeeded, so a target that cannot start fails the grant rather than
	// failing the first tunnel.
	Start(ctx context.Context, grant poolwire.LeaseGrant) (addr string, err error)
	// Stop tears the process down. It is called exactly once per started lease,
	// on release, on expiry, and on shutdown.
	Stop(leaseID string)
}

// errLeaseNotFound is returned for an unknown or already-released lease.
var errLeaseNotFound = errors.New("lease not found")

// leaseIDBytes is the entropy in a lease id. A lease id is a capability: whoever
// presents it on the tunnel endpoint, having already passed the cluster pin
// gate, is spliced to a tensor backend. Sixteen bytes makes guessing it
// irrelevant next to the pin check that has to pass first.
const leaseIDBytes = 16

// lease is one outstanding grant.
type lease struct {
	grant poolwire.LeaseGrant
	// holder is the node UUID the grant was issued to, taken from the
	// certificate the request arrived on rather than from anything in the body.
	// The tunnel endpoint checks it: a pinned cluster member is authorized to
	// ASK for a lease, which is not the same as being authorized to use one
	// somebody else holds.
	holder string
	// targetAddr is the loopback address the tunnel splices to.
	targetAddr string
	expiresAt  time.Time
	// conns tracks live tunnels so releasing a lease closes them rather than
	// leaving a stream spliced to a backend that is being torn down.
	conns map[uint64]func()
	nextC uint64
}

// LeaseStore is the donor side of pooling: which pools this node is currently
// holding memory for, and the loopback processes behind them.
//
// It is also the CommittedLedger the capacity collector reads, which is what
// keeps the two halves honest: memory promised to a lease stops being offered to
// the next planner in the same instant it is promised, rather than when the
// model finishes loading and the driver notices.
type LeaseStore struct {
	mu     sync.Mutex
	leases map[string]*lease
	target LoopbackTarget

	// now reads the clock. Overridden in tests.
	now func() time.Time
}

func NewLeaseStore(target LoopbackTarget) *LeaseStore {
	return &LeaseStore{
		leases: make(map[string]*lease),
		target: target,
		now:    time.Now,
	}
}

// CommittedBytes reports what live leases have claimed on one device, satisfying
// CommittedLedger. Expired leases are swept first, so memory returns as soon as
// a lease lapses rather than when somebody next asks about that lease.
func (s *LeaseStore) CommittedBytes(deviceIndex int) uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sweepLocked()

	var total uint64
	for _, l := range s.leases {
		for _, idx := range l.grant.DeviceIndexes {
			if idx == deviceIndex {
				// A lease naming several devices does not say how its bytes
				// divide between them, because the head has not decided yet
				// when it asks. Charging the whole figure to each named device
				// is the conservative reading, and being conservative about how
				// much memory is left is the only direction that cannot
				// overcommit a card.
				total += l.grant.Bytes
				break
			}
		}
	}
	return total
}

// Grant issues a lease to holder, or refuses with a reason the head can show a
// user.
//
// available is what the node could lend on each requested device right now,
// supplied by the caller rather than read here so the store never has to know
// how capacity is assembled — the collector already owns that, including this
// store's own commitments.
func (s *LeaseStore) Grant(ctx context.Context, holder string, req poolwire.LeaseRequest, available map[int]uint64) (poolwire.LeaseGrant, error) {
	var zero poolwire.LeaseGrant
	if holder == "" {
		return zero, errors.New("no holder identity")
	}
	if req.PoolID == "" {
		return zero, errors.New("no pool id")
	}
	if len(req.DeviceIndexes) == 0 {
		return zero, errors.New("no devices requested")
	}
	if req.Bytes == 0 {
		return zero, errors.New("no memory requested")
	}

	s.mu.Lock()
	s.sweepLocked()
	// One head at a time. rpc-server has no concept of sessions or of two
	// callers, and the memory a second pool planned against is memory the first
	// one is about to allocate. A donor that accepted both would fail both.
	for _, l := range s.leases {
		if l.grant.PoolID == req.PoolID && l.holder == holder {
			continue
		}
		s.mu.Unlock()
		return zero, fmt.Errorf("already leased to pool %s", l.grant.PoolID)
	}
	// A repeat request for a pool this node already serves is a renewal, not a
	// second lease: a head that retried through a dropped response must not
	// double-charge the device it is already using.
	for id, l := range s.leases {
		if l.grant.PoolID == req.PoolID && l.holder == holder {
			l.expiresAt = s.now().Add(req.TTL())
			l.grant.ExpiresAtMs = l.expiresAt.UnixMilli()
			grant := l.grant
			s.mu.Unlock()
			slog.Info("pool lease renewed by repeat request", "lease", id, "pool", req.PoolID)
			return grant, nil
		}
	}
	s.mu.Unlock()

	for _, idx := range req.DeviceIndexes {
		free, known := available[idx]
		if !known {
			return zero, fmt.Errorf("device %d is not available for pooling", idx)
		}
		if free < req.Bytes {
			return zero, fmt.Errorf("device %d has %d bytes free, needs %d", idx, free, req.Bytes)
		}
	}

	id, err := newLeaseID()
	if err != nil {
		return zero, fmt.Errorf("generate lease id: %w", err)
	}
	expires := s.now().Add(req.TTL())
	grant := poolwire.LeaseGrant{
		LeaseID:       id,
		PoolID:        req.PoolID,
		DeviceIndexes: append([]int(nil), req.DeviceIndexes...),
		Bytes:         req.Bytes,
		ExpiresAtMs:   expires.UnixMilli(),
	}

	// Start the local backend BEFORE recording the lease. A grant the head
	// believes in but whose process never came up is the worst failure available
	// here: the head plans a pool around memory that will never answer, and only
	// finds out when the model is already loading on every other node.
	addr, err := s.target.Start(ctx, grant)
	if err != nil {
		return zero, fmt.Errorf("start donor backend: %w", err)
	}

	s.mu.Lock()
	s.leases[id] = &lease{
		grant:      grant,
		holder:     holder,
		targetAddr: addr,
		expiresAt:  expires,
		conns:      make(map[uint64]func()),
	}
	s.mu.Unlock()

	slog.Info("pool lease granted", "lease", id, "pool", req.PoolID, "holder", holder,
		"devices", req.DeviceIndexes, "bytes", req.Bytes, "target", addr)
	return grant, nil
}

// Renew extends a lease held by holder.
func (s *LeaseStore) Renew(holder, leaseID string, ttl time.Duration) (poolwire.LeaseGrant, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sweepLocked()

	l, ok := s.leases[leaseID]
	if !ok || l.holder != holder {
		// A lease held by somebody else reads as absent rather than forbidden.
		// The distinction would tell a caller that a lease id it does not hold
		// exists, and there is nothing it could correctly do with that.
		return poolwire.LeaseGrant{}, errLeaseNotFound
	}
	l.expiresAt = s.now().Add(ttl)
	l.grant.ExpiresAtMs = l.expiresAt.UnixMilli()
	return l.grant, nil
}

// Release ends a lease held by holder, tearing down its tunnels and its backend.
func (s *LeaseStore) Release(holder, leaseID string) error {
	s.mu.Lock()
	l, ok := s.leases[leaseID]
	if !ok || l.holder != holder {
		s.mu.Unlock()
		return errLeaseNotFound
	}
	delete(s.leases, leaseID)
	closers := connClosers(l)
	s.mu.Unlock()

	// Close the streams first, then the process. The other order leaves a live
	// stream spliced to a backend that is shutting down, which surfaces to the
	// head as a truncated tensor transfer rather than as the orderly teardown it
	// asked for.
	for _, closeConn := range closers {
		closeConn()
	}
	s.target.Stop(leaseID)
	slog.Info("pool lease released", "lease", leaseID, "pool", l.grant.PoolID)
	return nil
}

// ReleaseAll ends every lease, for shutdown.
func (s *LeaseStore) ReleaseAll() {
	s.mu.Lock()
	all := make(map[string]*lease, len(s.leases))
	for id, l := range s.leases {
		all[id] = l
	}
	s.leases = make(map[string]*lease)
	s.mu.Unlock()

	for id, l := range all {
		for _, closeConn := range connClosers(l) {
			closeConn()
		}
		s.target.Stop(id)
	}
}

// Sweep drops expired leases and returns how many it dropped. It is called
// before every read of the store and on a timer, so a lapsed lease returns its
// memory whether or not anything asks about it.
func (s *LeaseStore) Sweep() int {
	s.mu.Lock()
	expired := s.expiredLocked()
	for id := range expired {
		delete(s.leases, id)
	}
	s.mu.Unlock()

	for id, l := range expired {
		for _, closeConn := range connClosers(l) {
			closeConn()
		}
		s.target.Stop(id)
		slog.Warn("pool lease expired without renewal; memory returned",
			"lease", id, "pool", l.grant.PoolID, "holder", l.holder)
	}
	return len(expired)
}

// sweepLocked drops expired leases while the caller holds the lock. Teardown of
// their backends is deferred to the next Sweep rather than performed here,
// because stopping a process under the store lock would block every capacity
// read on it.
func (s *LeaseStore) sweepLocked() {
	for id := range s.expiredLocked() {
		delete(s.leases, id)
	}
}

func (s *LeaseStore) expiredLocked() map[string]*lease {
	now := s.now()
	var out map[string]*lease
	for id, l := range s.leases {
		if now.After(l.expiresAt) {
			if out == nil {
				out = make(map[string]*lease)
			}
			out[id] = l
		}
	}
	return out
}

// connClosers snapshots a lease's tunnel closers.
func connClosers(l *lease) []func() {
	out := make([]func(), 0, len(l.conns))
	for _, closeConn := range l.conns {
		out = append(out, closeConn)
	}
	return out
}

// authorizeTunnel resolves a lease for a tunnel request and returns the loopback
// address to splice to.
//
// The holder check is the one that matters. Passing the cluster pin gate proves
// the caller is a member of this cluster, which authorizes it to ASK for a
// lease. It does not authorize it to use a lease another member holds, and
// without this check any pinned peer that learned a lease id would be spliced
// straight into a tensor backend somebody else is using.
func (s *LeaseStore) authorizeTunnel(holder, leaseID string) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sweepLocked()

	l, ok := s.leases[leaseID]
	if !ok || l.holder != holder {
		return "", errLeaseNotFound
	}
	return l.targetAddr, nil
}

// trackConn registers a live tunnel so releasing the lease closes it, returning
// a function that deregisters it. It reports false if the lease vanished between
// authorization and the stream being established.
func (s *LeaseStore) trackConn(leaseID string, closeConn func()) (func(), bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	l, ok := s.leases[leaseID]
	if !ok {
		return nil, false
	}
	l.nextC++
	id := l.nextC
	l.conns[id] = closeConn
	return func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		if l, ok := s.leases[leaseID]; ok {
			delete(l.conns, id)
		}
	}, true
}

// Snapshot returns the live leases, for status.
func (s *LeaseStore) Snapshot() []poolwire.LeaseGrant {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sweepLocked()
	out := make([]poolwire.LeaseGrant, 0, len(s.leases))
	for _, l := range s.leases {
		out = append(out, l.grant)
	}
	return out
}

// newLeaseID returns a fresh capability-grade identifier.
func newLeaseID() (string, error) {
	buf := make([]byte, leaseIDBytes)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return hex.EncodeToString(buf), nil
}
