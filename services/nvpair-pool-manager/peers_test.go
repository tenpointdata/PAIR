// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"path/filepath"
	"testing"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	"nvpair-shared/poolwire"
)

// pairedPool stands up two paired nodes where the second serves capacity,
// returning the first's mesh and the second's address.
func pairedPool(t *testing.T, inventory string, reserved uint64) (localMesh *clustertrust.Mesh, peerAddr string) {
	t.Helper()
	root := t.TempDir()
	localDir := filepath.Join(root, "local")
	peerDir := filepath.Join(root, "peer")
	clustertrusttest.Pair(t, "cluster-1", localDir, peerDir, "uuid-local", "uuid-peer")

	peerMesh := clustertrust.Open(peerDir)
	peerCollector := NewCollector(fakeNodeInfo(t, inventory), donorEnabled(t, reserved), nil, "peer-fallback")
	addr := startServer(t, peerMesh, peerCollector)

	return clustertrust.Open(localDir), addr
}

func TestCollectReadsAPeersCapacityOverMTLS(t *testing.T) {
	mesh, addr := pairedPool(t, twoGPUInventory, gb)
	pc := NewPeerCollector(mesh)
	pc.SetPeers([]poolwire.Peer{{NodeUUID: "uuid-peer", NodeName: "studio", Addresses: []string{addr}}})

	got := pc.Collect(context.Background())
	capacity, ok := got["uuid-peer"]
	if !ok {
		t.Fatalf("no capacity collected; got %+v", got)
	}
	if !capacity.DonorEnabled || len(capacity.Devices) != 2 {
		t.Fatalf("capacity = %+v, want an enabled donor with two devices", capacity)
	}
	// 32 - 4 used - 1 reserved, plus 24 - 1 used - 1 reserved.
	if want := (27 + 22) * gb; capacity.TotalFreeBytes() != want {
		t.Fatalf("TotalFreeBytes = %d, want %d", capacity.TotalFreeBytes(), want)
	}
}

// The reporting node names itself in the body, but the identity it is filed
// under is the one whose certificate pin was actually checked. Otherwise a peer
// could report capacity on another node's behalf.
func TestCollectedCapacityIsFiledUnderTheDialedIdentity(t *testing.T) {
	const impostor = `{"GPUs":[{"name":"X","vram_bytes":8589934592}],"telemetryValid":true,"hostUuid":"uuid-somebody-else"}`
	mesh, addr := pairedPool(t, impostor, 0)
	pc := NewPeerCollector(mesh)
	pc.SetPeers([]poolwire.Peer{{NodeUUID: "uuid-peer", Addresses: []string{addr}}})

	got := pc.Collect(context.Background())
	if _, wrong := got["uuid-somebody-else"]; wrong {
		t.Fatal("capacity was filed under the identity the body claimed")
	}
	capacity, ok := got["uuid-peer"]
	if !ok {
		t.Fatalf("capacity missing under the dialed identity; got %+v", got)
	}
	if capacity.NodeUUID != "uuid-peer" {
		t.Fatalf("NodeUUID = %q, want the dialed identity", capacity.NodeUUID)
	}
}

// A node that answers "no free memory" has been consulted and declined; one that
// did not answer has not been consulted. Collapsing the two would plan around a
// machine that is merely busy handshaking.
func TestUnreachablePeersAreOmittedNotReportedEmpty(t *testing.T) {
	mesh, addr := pairedPool(t, twoGPUInventory, 0)
	pc := NewPeerCollector(mesh)
	pc.SetPeers([]poolwire.Peer{
		{NodeUUID: "uuid-peer", Addresses: []string{addr}},
		{NodeUUID: "uuid-absent", Addresses: []string{"127.0.0.1:1"}},
	})

	got := pc.Collect(context.Background())
	if _, ok := got["uuid-peer"]; !ok {
		t.Fatal("the reachable peer should have answered")
	}
	if _, ok := got["uuid-absent"]; ok {
		t.Fatal("an unreachable peer must be absent, not present and empty")
	}
}

// A peer this node holds no pin for cannot be dialed at all — the client-side
// gate refuses before a connection is opened.
func TestUnpinnedPeersAreNeverDialed(t *testing.T) {
	mesh, addr := pairedPool(t, twoGPUInventory, 0)
	pc := NewPeerCollector(mesh)
	pc.SetPeers([]poolwire.Peer{{NodeUUID: "uuid-never-paired", Addresses: []string{addr}}})

	if got := pc.Collect(context.Background()); len(got) != 0 {
		t.Fatalf("collected %+v from an unpinned peer", got)
	}
}

// Capacity is a live measurement. A remembered reading for a departed member
// would let a planner build a pool around a machine that is gone.
func TestDepartedPeersAreForgotten(t *testing.T) {
	mesh, addr := pairedPool(t, twoGPUInventory, 0)
	pc := NewPeerCollector(mesh)
	pc.SetPeers([]poolwire.Peer{{NodeUUID: "uuid-peer", Addresses: []string{addr}}})

	if got := pc.Collect(context.Background()); len(got) != 1 {
		t.Fatalf("collected %d peers, want 1", len(got))
	}
	if len(pc.Last()) != 1 {
		t.Fatal("Last should hold the collection")
	}

	pc.SetPeers(nil)
	if got := pc.Last(); len(got) != 0 {
		t.Fatalf("Last still holds %+v after the peer left", got)
	}
}

func TestMalformedPeerEntriesAreIgnored(t *testing.T) {
	mesh, _ := pairedPool(t, twoGPUInventory, 0)
	pc := NewPeerCollector(mesh)
	pc.SetPeers([]poolwire.Peer{
		{NodeUUID: "", Addresses: []string{"127.0.0.1:9"}},
		{NodeUUID: "uuid-no-address"},
	})
	if got := pc.Collect(context.Background()); len(got) != 0 {
		t.Fatalf("collected %+v from unusable entries", got)
	}
}

// Last must not hand out the map the collector keeps mutating.
func TestLastIsACopy(t *testing.T) {
	mesh, addr := pairedPool(t, twoGPUInventory, 0)
	pc := NewPeerCollector(mesh)
	pc.SetPeers([]poolwire.Peer{{NodeUUID: "uuid-peer", Addresses: []string{addr}}})
	pc.Collect(context.Background())

	snap := pc.Last()
	delete(snap, "uuid-peer")
	snap["injected"] = poolwire.NodeCapacity{}

	again := pc.Last()
	if _, ok := again["uuid-peer"]; !ok {
		t.Fatal("mutating a snapshot removed a peer from the collector")
	}
	if _, ok := again["injected"]; ok {
		t.Fatal("mutating a snapshot added a peer to the collector")
	}
}
