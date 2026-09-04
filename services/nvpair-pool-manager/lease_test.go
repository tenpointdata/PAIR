// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"net"
	"sync"
	"testing"
	"time"

	"nvpair-shared/poolwire"
)

// echoTarget is a LoopbackTarget backed by an in-process echo listener, so the
// lease and tunnel machinery can be exercised without a llama.cpp build.
type echoTarget struct {
	t *testing.T

	mu       sync.Mutex
	started  map[string]net.Listener
	stopped  []string
	startErr error
}

func newEchoTarget(t *testing.T) *echoTarget {
	e := &echoTarget{t: t, started: map[string]net.Listener{}}
	t.Cleanup(func() {
		e.mu.Lock()
		defer e.mu.Unlock()
		for _, ln := range e.started {
			_ = ln.Close()
		}
	})
	return e
}

func (e *echoTarget) Start(_ context.Context, grant poolwire.LeaseGrant) (string, error) {
	e.mu.Lock()
	failWith := e.startErr
	e.mu.Unlock()
	if failWith != nil {
		return "", failWith
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", err
	}
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func() {
				defer func() { _ = conn.Close() }()
				buf := make([]byte, 32*1024)
				for {
					n, err := conn.Read(buf)
					if n > 0 {
						if _, werr := conn.Write(buf[:n]); werr != nil {
							return
						}
					}
					if err != nil {
						return
					}
				}
			}()
		}
	}()

	e.mu.Lock()
	e.started[grant.LeaseID] = ln
	e.mu.Unlock()
	return ln.Addr().String(), nil
}

func (e *echoTarget) Stop(leaseID string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if ln, ok := e.started[leaseID]; ok {
		_ = ln.Close()
		delete(e.started, leaseID)
	}
	e.stopped = append(e.stopped, leaseID)
}

func (e *echoTarget) stopCount() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.stopped)
}

func (e *echoTarget) failStarts(err error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.startErr = err
}

// leaseStore returns a store with a controllable clock.
func leaseStore(t *testing.T) (*LeaseStore, *echoTarget, func(time.Duration)) {
	t.Helper()
	now := time.Date(2026, 3, 1, 9, 0, 0, 0, time.UTC)
	target := newEchoTarget(t)
	s := NewLeaseStore(target)
	s.now = func() time.Time { return now }
	return s, target, func(d time.Duration) { now = now.Add(d) }
}

func plentiful() map[int]uint64 { return map[int]uint64{0: 32 * gb, 1: 32 * gb} }

func TestGrantValidatesItsRequest(t *testing.T) {
	s, _, _ := leaseStore(t)
	ctx := context.Background()
	base := poolwire.LeaseRequest{PoolID: "pool-1", DeviceIndexes: []int{0}, Bytes: gb}

	cases := map[string]struct {
		holder string
		req    poolwire.LeaseRequest
	}{
		"no holder":  {"", base},
		"no pool id": {"head", poolwire.LeaseRequest{DeviceIndexes: []int{0}, Bytes: gb}},
		"no devices": {"head", poolwire.LeaseRequest{PoolID: "pool-1", Bytes: gb}},
		"no bytes":   {"head", poolwire.LeaseRequest{PoolID: "pool-1", DeviceIndexes: []int{0}}},
	}
	for name, tc := range cases {
		if _, err := s.Grant(ctx, tc.holder, tc.req, plentiful()); err == nil {
			t.Errorf("%s should be refused", name)
		}
	}
}

func TestGrantRefusesMoreThanTheDeviceHas(t *testing.T) {
	s, _, _ := leaseStore(t)
	_, err := s.Grant(context.Background(), "head",
		poolwire.LeaseRequest{PoolID: "pool-1", DeviceIndexes: []int{0}, Bytes: 40 * gb},
		map[int]uint64{0: 8 * gb})
	if err == nil {
		t.Fatal("a request larger than the device should be refused")
	}
}

func TestGrantRefusesAnUnofferedDevice(t *testing.T) {
	s, _, _ := leaseStore(t)
	_, err := s.Grant(context.Background(), "head",
		poolwire.LeaseRequest{PoolID: "pool-1", DeviceIndexes: []int{7}, Bytes: gb},
		plentiful())
	if err == nil {
		t.Fatal("a device the node did not offer should be refused")
	}
}

// Memory promised to a lease stops being offered in the same instant it is
// promised, not when the model finishes loading and the driver notices.
func TestAGrantCommitsMemoryImmediately(t *testing.T) {
	s, _, _ := leaseStore(t)
	if got := s.CommittedBytes(0); got != 0 {
		t.Fatalf("committed = %d before any lease, want 0", got)
	}
	if _, err := s.Grant(context.Background(), "head",
		poolwire.LeaseRequest{PoolID: "pool-1", DeviceIndexes: []int{0}, Bytes: 12 * gb},
		plentiful()); err != nil {
		t.Fatalf("Grant: %v", err)
	}
	if got := s.CommittedBytes(0); got != 12*gb {
		t.Fatalf("committed = %d, want 12 GiB", got)
	}
	if got := s.CommittedBytes(1); got != 0 {
		t.Fatalf("an unnamed device shows %d committed, want 0", got)
	}
}

// rpc-server has no concept of two callers, and the memory a second pool plans
// against is memory the first is about to allocate.
func TestOnlyOnePoolAtATime(t *testing.T) {
	s, _, _ := leaseStore(t)
	ctx := context.Background()
	if _, err := s.Grant(ctx, "head-a",
		poolwire.LeaseRequest{PoolID: "pool-1", DeviceIndexes: []int{0}, Bytes: gb}, plentiful()); err != nil {
		t.Fatalf("first grant: %v", err)
	}
	if _, err := s.Grant(ctx, "head-b",
		poolwire.LeaseRequest{PoolID: "pool-2", DeviceIndexes: []int{1}, Bytes: gb}, plentiful()); err == nil {
		t.Fatal("a second pool should be refused even on a different device")
	}
}

// A head that retried through a dropped response must not double-charge the
// device it is already using.
func TestRepeatRequestForTheSamePoolRenews(t *testing.T) {
	s, target, advance := leaseStore(t)
	ctx := context.Background()
	req := poolwire.LeaseRequest{PoolID: "pool-1", DeviceIndexes: []int{0}, Bytes: 10 * gb}

	first, err := s.Grant(ctx, "head", req, plentiful())
	if err != nil {
		t.Fatalf("first grant: %v", err)
	}
	advance(20 * time.Second)
	second, err := s.Grant(ctx, "head", req, plentiful())
	if err != nil {
		t.Fatalf("repeat grant: %v", err)
	}
	if second.LeaseID != first.LeaseID {
		t.Fatalf("repeat produced a new lease %s, want the existing %s", second.LeaseID, first.LeaseID)
	}
	if second.ExpiresAtMs <= first.ExpiresAtMs {
		t.Fatal("a repeat request should extend the lease")
	}
	if got := s.CommittedBytes(0); got != 10*gb {
		t.Fatalf("committed = %d after a repeat, want the original 10 GiB", got)
	}
	if target.stopCount() != 0 {
		t.Fatal("a renewal should not have torn down the backend")
	}
}

// A grant the head believes in but whose backend never came up is the worst
// failure available: the head plans a pool around memory that will never answer.
func TestABackendThatCannotStartFailsTheGrant(t *testing.T) {
	s, target, _ := leaseStore(t)
	target.failStarts(errors.New("no such binary"))

	if _, err := s.Grant(context.Background(), "head",
		poolwire.LeaseRequest{PoolID: "pool-1", DeviceIndexes: []int{0}, Bytes: gb}, plentiful()); err == nil {
		t.Fatal("a grant should fail when its backend cannot start")
	}
	if got := s.CommittedBytes(0); got != 0 {
		t.Fatalf("committed = %d after a failed grant, want 0", got)
	}
	if len(s.Snapshot()) != 0 {
		t.Fatal("a failed grant must not leave a lease behind")
	}
}

// A lease id another member holds must read as absent rather than forbidden:
// the distinction would confirm the existence of a lease the caller has no
// business knowing about, and there is nothing it could correctly do with that.
func TestALeaseIsUsableOnlyByItsHolder(t *testing.T) {
	s, _, _ := leaseStore(t)
	grant, err := s.Grant(context.Background(), "head-a",
		poolwire.LeaseRequest{PoolID: "pool-1", DeviceIndexes: []int{0}, Bytes: gb}, plentiful())
	if err != nil {
		t.Fatalf("Grant: %v", err)
	}

	if _, err := s.Renew("head-b", grant.LeaseID, time.Minute); !errors.Is(err, errLeaseNotFound) {
		t.Fatalf("renew by another holder = %v, want not-found", err)
	}
	if err := s.Release("head-b", grant.LeaseID); !errors.Is(err, errLeaseNotFound) {
		t.Fatalf("release by another holder = %v, want not-found", err)
	}
	if _, err := s.authorizeTunnel("head-b", grant.LeaseID); !errors.Is(err, errLeaseNotFound) {
		t.Fatalf("tunnel by another holder = %v, want not-found", err)
	}
	if _, err := s.authorizeTunnel("head-a", grant.LeaseID); err != nil {
		t.Fatalf("the holder should be able to open its own tunnel: %v", err)
	}
}

// A head that dies must not strand a donor's memory forever. The lease lapses
// and the memory returns with no cleanup message from a process that is gone.
func TestAnUnrenewedLeaseLapsesAndReturnsItsMemory(t *testing.T) {
	s, target, advance := leaseStore(t)
	if _, err := s.Grant(context.Background(), "head",
		poolwire.LeaseRequest{PoolID: "pool-1", DeviceIndexes: []int{0}, Bytes: 16 * gb, TTLMs: 30_000},
		plentiful()); err != nil {
		t.Fatalf("Grant: %v", err)
	}
	if got := s.CommittedBytes(0); got != 16*gb {
		t.Fatalf("committed = %d, want 16 GiB", got)
	}

	advance(31 * time.Second)
	if swept := s.Sweep(); swept != 1 {
		t.Fatalf("swept %d leases, want 1", swept)
	}
	if got := s.CommittedBytes(0); got != 0 {
		t.Fatalf("committed = %d after expiry, want 0", got)
	}
	if target.stopCount() != 1 {
		t.Fatal("an expired lease should have torn down its backend")
	}
}

func TestRenewalKeepsALeaseAlive(t *testing.T) {
	s, _, advance := leaseStore(t)
	grant, err := s.Grant(context.Background(), "head",
		poolwire.LeaseRequest{PoolID: "pool-1", DeviceIndexes: []int{0}, Bytes: gb, TTLMs: 30_000},
		plentiful())
	if err != nil {
		t.Fatalf("Grant: %v", err)
	}
	for range 5 {
		advance(20 * time.Second)
		if _, err := s.Renew("head", grant.LeaseID, 30*time.Second); err != nil {
			t.Fatalf("Renew: %v", err)
		}
	}
	if s.Sweep() != 0 {
		t.Fatal("a renewed lease should not lapse")
	}
	if got := s.CommittedBytes(0); got != gb {
		t.Fatalf("committed = %d, want the lease still held", got)
	}
}

func TestReleaseFreesTheMemoryAndTheBackend(t *testing.T) {
	s, target, _ := leaseStore(t)
	grant, err := s.Grant(context.Background(), "head",
		poolwire.LeaseRequest{PoolID: "pool-1", DeviceIndexes: []int{0}, Bytes: gb}, plentiful())
	if err != nil {
		t.Fatalf("Grant: %v", err)
	}
	if err := s.Release("head", grant.LeaseID); err != nil {
		t.Fatalf("Release: %v", err)
	}
	if got := s.CommittedBytes(0); got != 0 {
		t.Fatalf("committed = %d after release, want 0", got)
	}
	if target.stopCount() != 1 {
		t.Fatal("release should have stopped the backend")
	}
	if err := s.Release("head", grant.LeaseID); !errors.Is(err, errLeaseNotFound) {
		t.Fatalf("second release = %v, want not-found", err)
	}
}

// A lease naming several devices does not say how its bytes divide between
// them, so each named device is charged the whole figure. Conservative about
// remaining memory is the only direction that cannot overcommit a card.
func TestAMultiDeviceLeaseChargesEachDevice(t *testing.T) {
	s, _, _ := leaseStore(t)
	if _, err := s.Grant(context.Background(), "head",
		poolwire.LeaseRequest{PoolID: "pool-1", DeviceIndexes: []int{0, 1}, Bytes: 8 * gb},
		plentiful()); err != nil {
		t.Fatalf("Grant: %v", err)
	}
	for _, idx := range []int{0, 1} {
		if got := s.CommittedBytes(idx); got != 8*gb {
			t.Fatalf("device %d committed = %d, want 8 GiB", idx, got)
		}
	}
}

func TestReleaseAllClearsEverything(t *testing.T) {
	s, target, _ := leaseStore(t)
	if _, err := s.Grant(context.Background(), "head",
		poolwire.LeaseRequest{PoolID: "pool-1", DeviceIndexes: []int{0}, Bytes: gb}, plentiful()); err != nil {
		t.Fatalf("Grant: %v", err)
	}
	s.ReleaseAll()
	if len(s.Snapshot()) != 0 {
		t.Fatal("ReleaseAll left a lease behind")
	}
	if target.stopCount() != 1 {
		t.Fatal("ReleaseAll should have stopped the backend")
	}
}

func TestLeaseTTLIsBounded(t *testing.T) {
	if got := (poolwire.LeaseRequest{}).TTL(); got != poolwire.DefaultLeaseTTL {
		t.Fatalf("unset TTL = %v, want the default", got)
	}
	if got := (poolwire.LeaseRequest{TTLMs: 1}).TTL(); got != poolwire.MinLeaseTTL {
		t.Fatalf("tiny TTL = %v, want the minimum", got)
	}
	if got := (poolwire.LeaseRequest{TTLMs: 86_400_000}).TTL(); got != poolwire.MaxLeaseTTL {
		t.Fatalf("huge TTL = %v, want the maximum", got)
	}
	if got := (poolwire.LeaseRequest{TTLMs: 45_000}).TTL(); got != 45*time.Second {
		t.Fatalf("in-range TTL = %v, want 45s", got)
	}
}
