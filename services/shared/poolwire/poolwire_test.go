// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package poolwire

import (
	"encoding/json"
	"testing"
	"time"
)

const gb = uint64(1) << 30

func TestFreeBytesSubtractsEveryClaim(t *testing.T) {
	d := DeviceCapacity{TotalBytes: 32 * gb, UsedBytes: 4 * gb, ReservedBytes: 2 * gb, CommittedBytes: 6 * gb}
	if got, want := d.FreeBytes(), 20*gb; got != want {
		t.Fatalf("FreeBytes = %d, want %d", got, want)
	}
}

// The three subtrahends come from independent sources — a driver reading, a user
// preference, and a lease ledger — and can briefly sum past the device's size.
// Wrapping unsigned would report the largest card in the cluster.
func TestFreeBytesSaturatesAtZero(t *testing.T) {
	for _, d := range []DeviceCapacity{
		{TotalBytes: 8 * gb, UsedBytes: 9 * gb},
		{TotalBytes: 8 * gb, ReservedBytes: 4 * gb, CommittedBytes: 5 * gb},
		{},
	} {
		if got := d.FreeBytes(); got != 0 {
			t.Fatalf("FreeBytes(%+v) = %d, want 0", d, got)
		}
	}
}

// Capacity nobody has agreed to lend is not capacity, however much of it there is.
func TestTotalFreeBytesIsZeroForANonDonor(t *testing.T) {
	n := NodeCapacity{
		DonorEnabled: false,
		Devices:      []DeviceCapacity{{TotalBytes: 32 * gb}, {TotalBytes: 32 * gb}},
	}
	if got := n.TotalFreeBytes(); got != 0 {
		t.Fatalf("TotalFreeBytes = %d for a non-donor, want 0", got)
	}
	n.DonorEnabled = true
	if got, want := n.TotalFreeBytes(), 64*gb; got != want {
		t.Fatalf("TotalFreeBytes = %d once lending, want %d", got, want)
	}
}

func TestObservedAtRendersTheTimestamp(t *testing.T) {
	when := time.Date(2026, 3, 1, 12, 0, 0, 0, time.UTC)
	n := NodeCapacity{ObservedAtMs: when.UnixMilli()}
	if got := n.ObservedAt(); !got.Equal(when) {
		t.Fatalf("ObservedAt = %v, want %v", got, when)
	}
}

// A lease that lapses too eagerly turns a busy head's missed renewal into a dead
// pool; one that lapses too slowly strands a donor's memory after a head crashes.
func TestLeaseTTLIsClampedToItsBounds(t *testing.T) {
	cases := map[string]struct {
		ms   int64
		want time.Duration
	}{
		"unset":        {0, DefaultLeaseTTL},
		"negative":     {-5, DefaultLeaseTTL},
		"below floor":  {1, MinLeaseTTL},
		"above cap":    {int64(MaxLeaseTTL/time.Millisecond) + 1, MaxLeaseTTL},
		"within range": {45_000, 45 * time.Second},
	}
	for name, tc := range cases {
		if got := (LeaseRequest{TTLMs: tc.ms}).TTL(); got != tc.want {
			t.Errorf("%s: TTL = %v, want %v", name, got, tc.want)
		}
	}
}

// The wire is what two independently-built binaries agree on, so the field names
// are pinned here rather than left to whatever a struct rename produces.
func TestCapacityFieldNamesAreStable(t *testing.T) {
	body, err := json.Marshal(NodeCapacity{
		NodeUUID:     "uuid",
		DonorEnabled: true,
		Devices:      []DeviceCapacity{{Index: 1, TotalBytes: gb, UsedBytes: 1, ReservedBytes: 2, CommittedBytes: 3}},
		ObservedAtMs: 7,
	})
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"nodeUuid", "donorEnabled", "devices", "observedAtMs", "telemetryValid"} {
		if _, ok := got[key]; !ok {
			t.Errorf("capacity is missing %q: %s", key, body)
		}
	}
	device, ok := got["devices"].([]any)
	if !ok || len(device) != 1 {
		t.Fatalf("devices did not survive: %s", body)
	}
	fields, ok := device[0].(map[string]any)
	if !ok {
		t.Fatalf("device is not an object: %s", body)
	}
	for _, key := range []string{"index", "totalBytes", "usedBytes", "reservedBytes", "committedBytes"} {
		if _, ok := fields[key]; !ok {
			t.Errorf("device is missing %q: %s", key, body)
		}
	}
}

// Paths are part of the contract too: a peer built from a different revision
// dials these strings.
func TestEndpointPathsAreStable(t *testing.T) {
	if CapacityPath != "/v1/pool/capacity" {
		t.Errorf("CapacityPath = %q", CapacityPath)
	}
	if LeasesPath != "/v1/pool/leases" {
		t.Errorf("LeasesPath = %q", LeasesPath)
	}
	if LeasePathPrefix != "/v1/pool/leases/" {
		t.Errorf("LeasePathPrefix = %q", LeasePathPrefix)
	}
	if TunnelPathPrefix != "/v1/pool/rpc/" {
		t.Errorf("TunnelPathPrefix = %q", TunnelPathPrefix)
	}
	if TunnelProtocol != "nvpair-ggml-rpc" {
		t.Errorf("TunnelProtocol = %q", TunnelProtocol)
	}
}
