// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"testing"

	"nvpair-shared/noderec"
	"nvpair-shared/poolwire"
	"nvpair-ui-broker/relay"
)

// poolNode builds a directory entry advertising the pool service.
func poolNode(uuid, name, ip string, ips []string, trusted bool) noderec.DirectoryNode {
	return noderec.DirectoryNode{
		HostUUID: uuid,
		Name:     name,
		IP:       ip,
		IPs:      ips,
		Trusted:  trusted,
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{
			noderec.ServicePool: {Port: poolManagerHTTPPort},
		},
	}
}

// brokerWith builds a broker holding a directory of the given nodes.
func brokerWith(selfUUID string, nodes ...noderec.DirectoryNode) *Broker {
	b := &Broker{nodeID: selfUUID, relayDir: relay.NewDirectory()}
	for _, n := range nodes {
		b.relayDir.Apply("discovery:node-discovered", n)
	}
	return b
}

func TestPoolPeersProjectsTheDirectory(t *testing.T) {
	b := brokerWith("uuid-self",
		poolNode("uuid-peer", "studio", "10.0.0.9", []string{"10.0.0.9", "192.168.5.20"}, true),
	)
	peers := b.poolPeers()
	if len(peers) != 1 {
		t.Fatalf("peers = %+v, want one", peers)
	}
	if peers[0].NodeUUID != "uuid-peer" || peers[0].NodeName != "studio" {
		t.Fatalf("peer identity = %+v", peers[0])
	}
	// The full ranked list, not just the canonical address: a multi-homed node
	// has no single address every peer can reach.
	want := []string{"10.0.0.9:14324", "192.168.5.20:14324"}
	if len(peers[0].Addresses) != len(want) {
		t.Fatalf("addresses = %v, want %v", peers[0].Addresses, want)
	}
	for i := range want {
		if peers[0].Addresses[i] != want[i] {
			t.Fatalf("addresses = %v, want %v", peers[0].Addresses, want)
		}
	}
}

// An untrusted node advertising the pool service is one this node holds no pin
// for, so every call to it fails the client-side gate before a connection is
// opened. Listing it would cost a pointless probe per sweep.
func TestPoolPeersExcludesUntrustedNodes(t *testing.T) {
	b := brokerWith("uuid-self",
		poolNode("uuid-stranger", "stranger", "10.0.0.7", nil, false),
		poolNode("uuid-peer", "peer", "10.0.0.9", nil, true),
	)
	peers := b.poolPeers()
	if len(peers) != 1 || peers[0].NodeUUID != "uuid-peer" {
		t.Fatalf("peers = %+v, want only the trusted node", peers)
	}
}

// The head's own capacity is read locally; listing itself as a peer would have
// it collect its own capacity over the network.
func TestPoolPeersExcludesThisNode(t *testing.T) {
	b := brokerWith("uuid-self", poolNode("uuid-self", "self", "10.0.0.5", nil, true))
	if peers := b.poolPeers(); len(peers) != 0 {
		t.Fatalf("peers = %+v, want none", peers)
	}
}

func TestPoolPeersSkipsNodesWithNoUsableEndpoint(t *testing.T) {
	noPort := poolNode("uuid-noport", "noport", "10.0.0.9", nil, true)
	noPort.Services[noderec.ServicePool] = noderec.ServiceStatus{Port: 0}

	noAddress := poolNode("uuid-noaddr", "noaddr", "", nil, true)

	b := brokerWith("uuid-self", noPort, noAddress)
	if peers := b.poolPeers(); len(peers) != 0 {
		t.Fatalf("peers = %+v, want none", peers)
	}
}

func TestPoolAddressesDeduplicates(t *testing.T) {
	n := poolNode("uuid-peer", "peer", "10.0.0.9", []string{"10.0.0.9", "10.0.0.9", "fd00::9"}, true)
	got := poolAddresses(n, 14324)
	want := []string{"10.0.0.9:14324", "[fd00::9]:14324"}
	if len(got) != len(want) {
		t.Fatalf("addresses = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("addresses = %v, want %v", got, want)
		}
	}
}

// An unchanged directory must produce no push, or the worker re-collects
// capacity from every peer because an unrelated node's GPU utilization moved.
func TestSamePoolPeersDetectsRealChangesOnly(t *testing.T) {
	a := []poolwire.Peer{{NodeUUID: "u1", NodeName: "n1", Addresses: []string{"10.0.0.1:14324"}}}
	same := []poolwire.Peer{{NodeUUID: "u1", NodeName: "n1", Addresses: []string{"10.0.0.1:14324"}}}
	if !samePoolPeers(a, same) {
		t.Fatal("identical sets should compare equal")
	}

	for name, changed := range map[string][]poolwire.Peer{
		"different address": {{NodeUUID: "u1", NodeName: "n1", Addresses: []string{"10.0.0.2:14324"}}},
		"different name":    {{NodeUUID: "u1", NodeName: "renamed", Addresses: []string{"10.0.0.1:14324"}}},
		"different uuid":    {{NodeUUID: "u2", NodeName: "n1", Addresses: []string{"10.0.0.1:14324"}}},
		"extra address":     {{NodeUUID: "u1", NodeName: "n1", Addresses: []string{"10.0.0.1:14324", "10.0.0.2:14324"}}},
		"empty":             {},
	} {
		if samePoolPeers(a, changed) {
			t.Errorf("%s should compare unequal", name)
		}
	}
}

// An unrecognized pool method must be a method-not-found from the broker rather
// than an unexplained silence from a worker that may not be running.
func TestPoolMethodsAreEnumeratedNotPrefixMatched(t *testing.T) {
	for _, method := range poolMethods {
		if !isPoolMethod(method) {
			t.Errorf("%s should be relayed to the pool manager", method)
		}
	}
	for _, method := range []string{"pool/unknown", "pool", "poolish/status", "engine:list"} {
		if isPoolMethod(method) {
			t.Errorf("%s must not be relayed", method)
		}
	}
}

// A request with no pool manager running gets an answer, not silence.
func TestRelayWithoutAPoolManagerAnswers(t *testing.T) {
	b := &Broker{}
	if got := b.getPoolManager(); got != nil {
		t.Fatal("a fresh broker should hold no pool manager")
	}
}
