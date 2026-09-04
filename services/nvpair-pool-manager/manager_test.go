// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"io"
	"path/filepath"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/poolwire"
)

// rig drives a Manager the way the broker does: JSON-RPC frames in one end,
// responses and notifications out the other.
type rig struct {
	t      *testing.T
	toMgr  *io.PipeWriter
	out    *Codec
	donor  *DonorState
	nextID int
}

// pipePair is one bidirectional pipe presented as a ReadWriter to each side.
type pipePair struct {
	io.Reader
	io.Writer
}

func newRig(t *testing.T, donor *DonorState, collector *Collector, peers *PeerCollector, leases *LeaseStore, pools *PoolManager) *rig {
	t.Helper()

	inR, inW := io.Pipe()   // test -> manager
	outR, outW := io.Pipe() // manager -> test

	mgr := NewManager(NewCodec(pipePair{Reader: inR, Writer: outW}), donor, collector, peers, leases, pools)
	ctx, cancel := context.WithCancel(context.Background())
	go func() { _ = mgr.Run(ctx) }()
	t.Cleanup(func() {
		cancel()
		_ = inW.Close()
		_ = outW.Close()
	})

	r := &rig{t: t, toMgr: inW, out: NewCodec(pipePair{Reader: outR, Writer: io.Discard}), donor: donor}
	// Every worker announces itself first; consume it so tests read their own
	// answers rather than this.
	if msg := r.read(); msg.Method != "ready" {
		t.Fatalf("first frame was %q, want ready", msg.Method)
	}
	return r
}

func (r *rig) read() *Message {
	r.t.Helper()
	type result struct {
		msg *Message
		err error
	}
	ch := make(chan result, 1)
	go func() {
		msg, err := r.out.Read()
		ch <- result{msg, err}
	}()
	select {
	case got := <-ch:
		if got.err != nil {
			r.t.Fatalf("read frame: %v", got.err)
		}
		return got.msg
	case <-time.After(5 * time.Second):
		r.t.Fatal("timed out waiting for a frame")
		return nil
	}
}

// call sends a request and returns the response frame.
func (r *rig) call(method string, params any) *Message {
	r.t.Helper()
	r.nextID++
	id, err := json.Marshal(r.nextID)
	if err != nil {
		r.t.Fatal(err)
	}
	raw := json.RawMessage(id)
	frame := Message{JSONRPC: "2.0", ID: &raw, Method: method}
	if params != nil {
		body, err := json.Marshal(params)
		if err != nil {
			r.t.Fatal(err)
		}
		frame.Params = body
	}
	line, err := json.Marshal(frame)
	if err != nil {
		r.t.Fatal(err)
	}
	if _, err := r.toMgr.Write(append(line, '\n')); err != nil {
		r.t.Fatalf("write frame: %v", err)
	}
	return r.read()
}

func decodeStatus(t *testing.T, msg *Message) statusResult {
	t.Helper()
	var got statusResult
	if err := json.Unmarshal(msg.Result, &got); err != nil {
		t.Fatalf("decode status: %v (result %s)", err, msg.Result)
	}
	return got
}

func newTestRig(t *testing.T) *rig {
	t.Helper()
	donor := &DonorState{}
	collector := NewCollector(fakeNodeInfo(t, twoGPUInventory), donor, nil, "uuid-local")
	peers := NewPeerCollector(clustertrust.Open(filepath.Join(t.TempDir(), "cluster")))
	return newRig(t, donor, collector, peers, nil, nil)
}

func TestStatusReportsLocalCapacityAndPolicy(t *testing.T) {
	r := newTestRig(t)
	got := decodeStatus(t, r.call(poolwire.MethodStatus, nil))

	if got.Donor.Enabled {
		t.Fatal("a fresh node must not be a donor")
	}
	if len(got.Local.Devices) != 2 {
		t.Fatalf("local devices = %d, want 2", len(got.Local.Devices))
	}
	if got.ClusterFreeBytes != 0 {
		t.Fatalf("ClusterFreeBytes = %d for a non-donor, want 0", got.ClusterFreeBytes)
	}
}

func TestEnablingDonationChangesTheOfferedCapacity(t *testing.T) {
	r := newTestRig(t)

	enabled := true
	resp := r.call(poolwire.MethodSetDonor, map[string]any{"enabled": enabled})
	if resp.Error != nil {
		t.Fatalf("set donor: %v", resp.Error)
	}

	// The change is announced after the response, so a caller's completion
	// callback always observes the notification strictly later than its answer.
	notif := r.read()
	if notif.Method != poolwire.NotifyUpdated {
		t.Fatalf("expected %s after the response, got %q", poolwire.NotifyUpdated, notif.Method)
	}

	got := decodeStatus(t, r.call(poolwire.MethodStatus, nil))
	if !got.Donor.Enabled {
		t.Fatal("donation should be on")
	}
	if want := (28 + 23) * gb; got.ClusterFreeBytes != want {
		t.Fatalf("ClusterFreeBytes = %d, want %d", got.ClusterFreeBytes, want)
	}
}

// A user toggling donation off should not have to restate a reservation they set
// months ago, and a caller that omits a field must not reset it to zero.
func TestDonorFieldsAreIndependentlySettable(t *testing.T) {
	r := newTestRig(t)

	if resp := r.call(poolwire.MethodSetDonor, map[string]any{"enabled": true, "reservedBytes": 3 * gb}); resp.Error != nil {
		t.Fatalf("set both: %v", resp.Error)
	}
	r.read() // the update notification

	if resp := r.call(poolwire.MethodSetDonor, map[string]any{"enabled": false}); resp.Error != nil {
		t.Fatalf("set enabled only: %v", resp.Error)
	}
	r.read()

	got := r.donor.Settings()
	if got.Enabled {
		t.Fatal("donation should be off")
	}
	if got.ReservedBytes != 3*gb {
		t.Fatalf("ReservedBytes = %d, want the 3 GiB that was not restated", got.ReservedBytes)
	}
}

func TestNoOpDonorSetIsNotAnnounced(t *testing.T) {
	r := newTestRig(t)

	if resp := r.call(poolwire.MethodSetDonor, map[string]any{"enabled": true}); resp.Error != nil {
		t.Fatalf("first set: %v", resp.Error)
	}
	if notif := r.read(); notif.Method != poolwire.NotifyUpdated {
		t.Fatalf("expected an update notification, got %q", notif.Method)
	}

	// Re-asserting the same value answers but announces nothing, so the next
	// frame must be the response to a different call rather than a notification.
	if resp := r.call(poolwire.MethodSetDonor, map[string]any{"enabled": true}); resp.Error != nil {
		t.Fatalf("second set: %v", resp.Error)
	}
	next := r.call(poolwire.MethodStatus, nil)
	if next.Method != "" {
		t.Fatalf("a no-op set announced %q", next.Method)
	}
}

func TestDonorSetRejectsUnusableParams(t *testing.T) {
	r := newTestRig(t)

	if resp := r.call(poolwire.MethodSetDonor, map[string]any{}); resp.Error == nil {
		t.Fatal("a set with nothing to change should be rejected")
	}
	if resp := r.call(poolwire.MethodSetDonor, map[string]any{"reservedBytes": maxReservedBytes + 1}); resp.Error == nil {
		t.Fatal("an implausible reservation should be rejected")
	}
	if r.donor.Settings().ReservedBytes != 0 {
		t.Fatal("a rejected set must not have applied")
	}
}

func TestSetPeersIsAccepted(t *testing.T) {
	r := newTestRig(t)
	resp := r.call(poolwire.MethodSetPeers, map[string]any{
		"peers": []poolwire.Peer{{NodeUUID: "uuid-peer", Addresses: []string{"10.0.0.9:14324"}}},
	})
	if resp.Error != nil {
		t.Fatalf("set peers: %v", resp.Error)
	}
	var got struct {
		OK    bool `json:"ok"`
		Peers int  `json:"peers"`
	}
	if err := json.Unmarshal(resp.Result, &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !got.OK || got.Peers != 1 {
		t.Fatalf("result = %+v, want one peer accepted", got)
	}
}

func TestUnknownMethodIsRefused(t *testing.T) {
	r := newTestRig(t)
	resp := r.call("pool/does-not-exist", nil)
	if resp.Error == nil || resp.Error.Code != -32601 {
		t.Fatalf("error = %+v, want method-not-found", resp.Error)
	}
}

// A node that has enabled donation but cannot run a backend is a different
// problem from one that has declined. Collapsing them hides a misconfiguration
// behind a switch that looks correct.
func TestStatusDistinguishesCannotDonateFromWillNot(t *testing.T) {
	donor := &DonorState{}
	collector := NewCollector(fakeNodeInfo(t, twoGPUInventory), donor, nil, "uuid-local")
	peers := NewPeerCollector(clustertrust.Open(filepath.Join(t.TempDir(), "cluster")))

	unable := newRig(t, donor, collector, peers, nil, nil)
	if got := decodeStatus(t, unable.call(poolwire.MethodStatus, nil)); got.DonorReady {
		t.Fatal("a node with no backend command should not report itself ready to donate")
	}

	target := newEchoTarget(t)
	able := newRig(t, donor, collector, peers, NewLeaseStore(target), nil)
	if got := decodeStatus(t, able.call(poolwire.MethodStatus, nil)); !got.DonorReady {
		t.Fatal("a node with a backend command should report itself ready")
	}
}

// "My GPU is full and I did not start anything" is otherwise unanswerable from
// the interface.
func TestStatusReportsOutstandingLeases(t *testing.T) {
	donor := &DonorState{}
	leases := NewLeaseStore(newEchoTarget(t))
	collector := NewCollector(fakeNodeInfo(t, twoGPUInventory), donor, leases, "uuid-local")
	peers := NewPeerCollector(clustertrust.Open(filepath.Join(t.TempDir(), "cluster")))
	r := newRig(t, donor, collector, peers, leases, nil)

	if _, err := leases.Grant(context.Background(), "uuid-head",
		poolwire.LeaseRequest{PoolID: "pool-1", DeviceIndexes: []int{0}, Bytes: 6 * gb},
		map[int]uint64{0: 32 * gb}); err != nil {
		t.Fatalf("Grant: %v", err)
	}

	got := decodeStatus(t, r.call(poolwire.MethodStatus, nil))
	if len(got.Leases) != 1 || got.Leases[0].PoolID != "pool-1" {
		t.Fatalf("leases = %+v, want the outstanding grant", got.Leases)
	}
	// The committed memory is subtracted from what this node still offers.
	if got.Local.Devices[0].CommittedBytes != 6*gb {
		t.Fatalf("device 0 committed = %d, want 6 GiB", got.Local.Devices[0].CommittedBytes)
	}
}
