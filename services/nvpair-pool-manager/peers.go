// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/poolwire"
	"nvpair-shared/reach"
)

// peerRequestTimeout bounds one peer capacity call end to end.
const peerRequestTimeout = 5 * time.Second

// peerFanoutLimit bounds how many peers are asked at once. A pool cluster is a
// handful of machines, so this is not about scale — it is so a sweep against a
// dozen unreachable peers opens a dozen sockets rather than however many the
// directory happens to hold.
const peerFanoutLimit = 8

// capacityMaxBytes caps a peer's capacity response. The document is a few
// hundred bytes per device; the cap is here because the body arrives from the
// network and an unbounded read is an unbounded allocation.
const capacityMaxBytes = 1 << 20

// PeerCollector fetches capacity from the cluster's other members.
//
// It does not discover them. The peer set is pushed in by the broker, which
// already holds the consolidated discovery directory — a second mDNS browser in
// this process would compete for the same multicast socket to learn something
// the parent already knows.
type PeerCollector struct {
	mesh    *clustertrust.Mesh
	clients *clustertrust.PeerClientPool
	chooser *reach.Chooser

	mu    sync.RWMutex
	peers map[string]poolwire.Peer
	last  map[string]poolwire.NodeCapacity
}

func NewPeerCollector(mesh *clustertrust.Mesh) *PeerCollector {
	return &PeerCollector{
		mesh:    mesh,
		clients: clustertrust.NewPeerClientPool(mesh, peerRequestTimeout),
		chooser: reach.NewChooser(),
		peers:   make(map[string]poolwire.Peer),
		last:    make(map[string]poolwire.NodeCapacity),
	}
}

// SetPeers replaces the set of members to collect from.
//
// Replacing rather than merging is correct here, unlike in an address book: this
// is the broker's current statement of who is in the cluster, and a member that
// has left must stop being asked. Capacity readings for departed peers are
// dropped with them, because capacity is a live measurement and a remembered one
// would let a planner build a pool around a machine that is gone.
func (p *PeerCollector) SetPeers(peers []poolwire.Peer) {
	next := make(map[string]poolwire.Peer, len(peers))
	for _, peer := range peers {
		if peer.NodeUUID == "" || len(peer.Addresses) == 0 {
			continue
		}
		next[peer.NodeUUID] = peer
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.peers = next
	for uuid := range p.last {
		if _, still := next[uuid]; !still {
			delete(p.last, uuid)
			p.chooser.Forget(uuid)
		}
	}
}

// Collect asks every known peer for its capacity and returns what answered.
//
// A peer that fails is omitted rather than reported as empty. The distinction
// matters to a planner: a node that answers "no free memory" has been consulted
// and declined, while a node that did not answer has not been consulted, and
// treating the second as the first would quietly plan around a machine that is
// merely busy handshaking.
func (p *PeerCollector) Collect(ctx context.Context) map[string]poolwire.NodeCapacity {
	p.mu.RLock()
	targets := make([]poolwire.Peer, 0, len(p.peers))
	for _, peer := range p.peers {
		targets = append(targets, peer)
	}
	p.mu.RUnlock()

	p.mesh.Refresh()

	results := make(chan poolwire.NodeCapacity, len(targets))
	sem := make(chan struct{}, peerFanoutLimit)
	var wg sync.WaitGroup

	for _, peer := range targets {
		wg.Add(1)
		go func(peer poolwire.Peer) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			capacity, err := p.fetch(ctx, peer)
			if err != nil {
				slog.Debug("pool: peer capacity unavailable", "peer", peer.NodeUUID, "err", err)
				return
			}
			results <- capacity
		}(peer)
	}
	wg.Wait()
	close(results)

	out := make(map[string]poolwire.NodeCapacity, len(targets))
	for capacity := range results {
		// The reporting node names itself in the body. Trusting that over the
		// UUID we dialed would let a peer report capacity on another node's
		// behalf, so the dialed identity — the one the certificate pin was
		// checked against — is the one it is filed under.
		out[capacity.NodeUUID] = capacity
	}

	p.mu.Lock()
	p.last = out
	p.mu.Unlock()
	return out
}

// Last returns the most recent successful collection without re-fetching, for
// a caller that wants the current view rather than a fresh one.
func (p *PeerCollector) Last() map[string]poolwire.NodeCapacity {
	p.mu.RLock()
	defer p.mu.RUnlock()
	out := make(map[string]poolwire.NodeCapacity, len(p.last))
	for uuid, capacity := range p.last {
		out[uuid] = capacity
	}
	return out
}

// fetch reads one peer's capacity over cluster mTLS.
func (p *PeerCollector) fetch(ctx context.Context, peer poolwire.Peer) (poolwire.NodeCapacity, error) {
	var out poolwire.NodeCapacity

	client, ok := p.clients.Client(peer.NodeUUID)
	if !ok {
		return out, fmt.Errorf("no pin for peer")
	}
	// Confirm which of a multi-homed peer's addresses actually answers before
	// spending a request timeout on one that never will. The chooser remembers,
	// so this costs a probe only when the previous answer stops working.
	addr := p.chooser.ChooseWithin(ctx, peer.NodeUUID, peer.Addresses)
	if addr == "" {
		return out, fmt.Errorf("no reachable address")
	}

	url := "https://" + addr + poolwire.CapacityPath
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return out, fmt.Errorf("build request: %w", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		return out, fmt.Errorf("get capacity: %w", err)
	}
	defer func() { _, _ = io.Copy(io.Discard, resp.Body); _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return out, fmt.Errorf("peer returned %s", resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, capacityMaxBytes))
	if err != nil {
		return out, fmt.Errorf("read body: %w", err)
	}
	if err := json.Unmarshal(body, &out); err != nil {
		return out, fmt.Errorf("parse capacity: %w", err)
	}
	out.NodeUUID = peer.NodeUUID
	if out.NodeName == "" {
		out.NodeName = peer.NodeName
	}
	return out, nil
}
