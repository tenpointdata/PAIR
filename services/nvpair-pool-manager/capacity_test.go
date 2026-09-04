// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"nvpair-shared/poolwire"
)

const gb = uint64(1) << 30

// fakeNodeInfo stands in for the local inventory service. The body is the real
// wire shape, spelled out rather than built from a struct, so a rename of a JSON
// tag on either side fails this test instead of silently reading as zero.
func fakeNodeInfo(t *testing.T, body string) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

const twoGPUInventory = `{
  "GPUs": [
    {"name": "NVIDIA GeForce RTX 5090", "vram_bytes": 34359738368, "vram_used_bytes": 4294967296},
    {"name": "NVIDIA GeForce RTX 4090", "vram_bytes": 25769803776, "vram_used_bytes": 1073741824}
  ],
  "telemetryValid": true,
  "hostUuid": "host-from-node-info"
}`

func donorEnabled(t *testing.T, reserved uint64) *DonorState {
	t.Helper()
	s := &DonorState{}
	if _, err := s.Set(poolwire.DonorSettings{Enabled: true, ReservedBytes: reserved}); err != nil {
		t.Fatal(err)
	}
	return s
}

func TestLocalCapacityReadsTheInventory(t *testing.T) {
	c := NewCollector(fakeNodeInfo(t, twoGPUInventory), donorEnabled(t, 0), nil, "fallback-uuid")
	got := c.Local(context.Background())

	if !got.DonorEnabled {
		t.Fatal("donor should be enabled")
	}
	if got.NodeUUID != "host-from-node-info" {
		t.Fatalf("NodeUUID = %q; node-info's hostUuid is authoritative over the fallback", got.NodeUUID)
	}
	if !got.TelemetryValid {
		t.Fatal("TelemetryValid should carry through")
	}
	if len(got.Devices) != 2 {
		t.Fatalf("devices = %d, want 2", len(got.Devices))
	}
	if got.Devices[0].Index != 0 || got.Devices[1].Index != 1 {
		t.Fatalf("device indexes = %d,%d, want 0,1", got.Devices[0].Index, got.Devices[1].Index)
	}
	if got.Devices[0].TotalBytes != 32*gb || got.Devices[0].UsedBytes != 4*gb {
		t.Fatalf("device 0 = %+v, want 32 GiB total / 4 GiB used", got.Devices[0])
	}
	// 32 - 4 free, plus 24 - 1 free.
	if want := (28 + 23) * gb; got.TotalFreeBytes() != want {
		t.Fatalf("TotalFreeBytes = %d, want %d", got.TotalFreeBytes(), want)
	}
}

// The owner's reservation is per device, because it exists to protect
// interactive use and each display runs on one card.
func TestReservationAppliesToEveryDevice(t *testing.T) {
	c := NewCollector(fakeNodeInfo(t, twoGPUInventory), donorEnabled(t, 2*gb), nil, "fallback")
	got := c.Local(context.Background())

	for _, d := range got.Devices {
		if d.ReservedBytes != 2*gb {
			t.Fatalf("device %d reserved %d, want 2 GiB", d.Index, d.ReservedBytes)
		}
	}
	if want := (26 + 21) * gb; got.TotalFreeBytes() != want {
		t.Fatalf("TotalFreeBytes = %d, want %d", got.TotalFreeBytes(), want)
	}
}

// fixedLedger reports a commitment on one device.
type fixedLedger map[int]uint64

func (f fixedLedger) CommittedBytes(index int) uint64 { return f[index] }

// A lease is a promise made before the memory is touched. Until the model
// finishes loading the driver still reports it free, and a second pool planned
// against it would overcommit the device.
func TestCommittedBytesAreSubtractedBeforeTheDriverSeesThem(t *testing.T) {
	c := NewCollector(fakeNodeInfo(t, twoGPUInventory), donorEnabled(t, 0), fixedLedger{0: 20 * gb}, "fallback")
	got := c.Local(context.Background())

	if got.Devices[0].CommittedBytes != 20*gb {
		t.Fatalf("committed = %d, want 20 GiB", got.Devices[0].CommittedBytes)
	}
	if free := got.Devices[0].FreeBytes(); free != 8*gb {
		t.Fatalf("free after commitment = %d, want 8 GiB", free)
	}
	if got.Devices[1].CommittedBytes != 0 {
		t.Fatal("an uncommitted device must not inherit another's commitment")
	}
}

// The three subtrahends come from independent sources and can briefly sum past
// the device's size. Wrapping unsigned would report the largest card in the
// cluster.
func TestFreeBytesSaturatesRatherThanWrapping(t *testing.T) {
	d := poolwire.DeviceCapacity{TotalBytes: 8 * gb, UsedBytes: 6 * gb, ReservedBytes: 4 * gb, CommittedBytes: 2 * gb}
	if got := d.FreeBytes(); got != 0 {
		t.Fatalf("FreeBytes = %d, want 0", got)
	}
}

// Capacity nobody agreed to lend is not capacity, however much of it there is.
func TestDisabledDonorOffersNothing(t *testing.T) {
	c := NewCollector(fakeNodeInfo(t, twoGPUInventory), &DonorState{}, nil, "fallback")
	got := c.Local(context.Background())

	if got.DonorEnabled {
		t.Fatal("donor should be disabled")
	}
	if got.TotalFreeBytes() != 0 {
		t.Fatalf("TotalFreeBytes = %d for a non-donor, want 0", got.TotalFreeBytes())
	}
	if len(got.Devices) == 0 {
		t.Fatal("the devices should still be reported, so a user can see what enabling would offer")
	}
}

// A device whose total is unknown would read as a card claiming to have nothing.
func TestDevicesWithNoReportedVRAMAreOmitted(t *testing.T) {
	const body = `{"GPUs":[{"name":"Integrated"},{"name":"Real","vram_bytes":8589934592}],"telemetryValid":true}`
	c := NewCollector(fakeNodeInfo(t, body), donorEnabled(t, 0), nil, "fallback")
	got := c.Local(context.Background())

	if len(got.Devices) != 1 || got.Devices[0].Name != "Real" {
		t.Fatalf("devices = %+v, want only the one reporting VRAM", got.Devices)
	}
	// The surviving device keeps its true inventory index, so a lease naming it
	// selects the same card the node would.
	if got.Devices[0].Index != 1 {
		t.Fatalf("device index = %d, want its inventory position 1", got.Devices[0].Index)
	}
}

// A node whose node-info is restarting is still a member with a knowable policy.
// Reporting "enabled, no devices" tells a planner to look elsewhere this round;
// failing would make a sibling restart look like an unreachable peer.
func TestUnreachableInventoryStillReportsPolicy(t *testing.T) {
	c := NewCollector("http://127.0.0.1:1/v1/node-info", donorEnabled(t, 0), nil, "fallback-uuid")
	got := c.Local(context.Background())

	if !got.DonorEnabled {
		t.Fatal("the donation policy is known even when the inventory is not")
	}
	if got.NodeUUID != "fallback-uuid" {
		t.Fatalf("NodeUUID = %q, want the fallback when node-info cannot answer", got.NodeUUID)
	}
	if len(got.Devices) != 0 {
		t.Fatal("no inventory means no devices, not invented ones")
	}
	if got.ObservedAtMs == 0 {
		t.Fatal("the reading should still be timestamped")
	}
}

func TestInventoryErrorsAreNotFatal(t *testing.T) {
	for _, body := range []string{"not json at all", `{"GPUs": "wrong type"}`} {
		c := NewCollector(fakeNodeInfo(t, body), donorEnabled(t, 0), nil, "fallback")
		got := c.Local(context.Background())
		if !got.DonorEnabled || len(got.Devices) != 0 {
			t.Fatalf("body %q produced %+v, want policy with no devices", body, got)
		}
	}
}

func TestInventoryNon200IsTreatedAsUnavailable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "starting", http.StatusServiceUnavailable)
	}))
	t.Cleanup(srv.Close)

	c := NewCollector(srv.URL, donorEnabled(t, 0), nil, "fallback")
	if got := c.Local(context.Background()); len(got.Devices) != 0 {
		t.Fatalf("devices = %+v, want none", got.Devices)
	}
}

// The reported identity starts as the local fallback and becomes node-info's
// hostUuid once it has answered, so a peer keys capacity by the identity the
// scanner already advertises for this node.
func TestReportedIdentityAdoptsNodeInfosHostUUID(t *testing.T) {
	c := NewCollector(fakeNodeInfo(t, twoGPUInventory), donorEnabled(t, 0), nil, "fallback-uuid")
	if got := c.NodeUUID(); got != "fallback-uuid" {
		t.Fatalf("NodeUUID before any reading = %q, want the fallback", got)
	}

	c.Local(context.Background())
	if got := c.NodeUUID(); got != "host-from-node-info" {
		t.Fatalf("NodeUUID after a reading = %q, want node-info's hostUuid", got)
	}
}

// Local runs on every peer request, and those are concurrent.
func TestConcurrentCapacityReadsAreSafe(t *testing.T) {
	c := NewCollector(fakeNodeInfo(t, twoGPUInventory), donorEnabled(t, gb), nil, "fallback")
	done := make(chan struct{})
	for range 8 {
		go func() {
			defer func() { done <- struct{}{} }()
			for range 20 {
				if got := c.Local(context.Background()); len(got.Devices) != 2 {
					t.Errorf("devices = %d, want 2", len(got.Devices))
					return
				}
				_ = c.NodeUUID()
			}
		}()
	}
	for range 8 {
		<-done
	}
}
