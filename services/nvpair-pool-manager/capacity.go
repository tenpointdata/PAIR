// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"time"

	"nvpair-shared/poolwire"
)

// nodeInfoTimeout bounds the local inventory read. It is generous for a
// loopback call and finite because node-info is a supervised sibling that can be
// restarting, and a capacity request that blocks on it would hold up a peer's.
const nodeInfoTimeout = 3 * time.Second

// nodeInfoMaxBytes caps the inventory response. The real payload is a few
// kilobytes; the cap exists because this is parsed from a socket and an
// unbounded read from any socket is an unbounded allocation.
const nodeInfoMaxBytes = 1 << 20

// nodeInfoGPU is the subset of nvpair-node-info's per-GPU payload this service
// reads. Declared here rather than imported because node-info is a package main
// in another module; the field tags are the contract, and they are checked by a
// test against a captured response.
type nodeInfoGPU struct {
	Name          string `json:"name"`
	VramBytes     uint64 `json:"vram_bytes"`
	VramUsedBytes uint64 `json:"vram_used_bytes"`
}

// nodeInfoResponse is the subset of the inventory document this service reads.
type nodeInfoResponse struct {
	GPUs           []nodeInfoGPU `json:"GPUs"`
	TelemetryValid bool          `json:"telemetryValid"`
	HostUUID       string        `json:"hostUuid"`
}

// CommittedLedger reports how many bytes outstanding leases have already
// claimed on one of this node's devices.
//
// It is an interface so the capacity collector can be tested without a lease
// store, and because the two have genuinely different lifetimes: capacity is
// read on every peer request, while commitments change only when a pool forms
// or ends.
type CommittedLedger interface {
	CommittedBytes(deviceIndex int) uint64
}

// noCommitments is the ledger used before any lease machinery exists, and by
// tests that are not about leases.
type noCommitments struct{}

func (noCommitments) CommittedBytes(int) uint64 { return 0 }

// Collector assembles this node's poolable capacity from the local hardware
// inventory, the owner's donation policy, and the lease ledger.
//
// The three sources are combined here and nowhere else. Each is authoritative
// about a different thing and none of them can answer alone: node-info knows
// what the driver reports, only this machine's owner knows what they are willing
// to lend, and only the lease store knows what has already been promised.
type Collector struct {
	nodeInfoURL string
	donor       *DonorState
	ledger      CommittedLedger
	client      *http.Client

	// nodeName identifies this node in every capacity report, for display.
	nodeName string

	// mu guards nodeUUID, which starts as the locally-derived fallback and is
	// replaced the first time node-info answers. Guarded because Local runs on
	// every peer request and those are concurrent.
	mu sync.RWMutex
	// nodeUUID is the identity this node reports under. node-info's hostUuid is
	// preferred once known — it is the same value the scanner advertises, so a
	// peer keys capacity by the identity it already holds — and the fallback
	// covers the window before node-info has answered.
	nodeUUID string
}

// NewCollector builds a collector reading the local node-info service.
func NewCollector(nodeInfoURL string, donor *DonorState, ledger CommittedLedger, fallbackNodeUUID string) *Collector {
	if ledger == nil {
		ledger = noCommitments{}
	}
	name, err := os.Hostname()
	if err != nil {
		name = ""
	}
	return &Collector{
		nodeInfoURL: nodeInfoURL,
		donor:       donor,
		ledger:      ledger,
		client:      &http.Client{Timeout: nodeInfoTimeout},
		nodeUUID:    fallbackNodeUUID,
		nodeName:    name,
	}
}

// Local returns this node's capacity right now.
//
// An inventory read that fails is not an error here. A node whose node-info is
// restarting is still a cluster member with a knowable donation policy, and
// reporting "donor enabled, no devices" is both true and useful — it tells a
// planner to look elsewhere for this round rather than to treat the node as
// absent. Returning an error would instead make a transient sibling restart
// look like an unreachable peer.
func (c *Collector) Local(ctx context.Context) poolwire.NodeCapacity {
	settings := c.donor.Settings()
	out := poolwire.NodeCapacity{
		NodeUUID:     c.NodeUUID(),
		NodeName:     c.nodeName,
		DonorEnabled: settings.Enabled,
		ObservedAtMs: time.Now().UnixMilli(),
	}

	info, err := c.readNodeInfo(ctx)
	if err != nil {
		return out
	}
	if info.HostUUID != "" {
		out.NodeUUID = info.HostUUID
		c.mu.Lock()
		c.nodeUUID = info.HostUUID
		c.mu.Unlock()
	}
	out.TelemetryValid = info.TelemetryValid

	for i, gpu := range info.GPUs {
		// A device whose total is unknown is skipped rather than reported with a
		// zero total. Zero would read as "no memory", which a planner would
		// treat as a device it can never use — the same outcome — but it would
		// also show a user a card claiming to have nothing, which invites a bug
		// report about the wrong thing.
		if gpu.VramBytes == 0 {
			continue
		}
		out.Devices = append(out.Devices, poolwire.DeviceCapacity{
			Index:          i,
			Name:           gpu.Name,
			TotalBytes:     gpu.VramBytes,
			UsedBytes:      gpu.VramUsedBytes,
			ReservedBytes:  settings.ReservedBytes,
			CommittedBytes: c.ledger.CommittedBytes(i),
		})
	}
	return out
}

// readNodeInfo fetches and parses the local inventory.
func (c *Collector) readNodeInfo(ctx context.Context) (nodeInfoResponse, error) {
	var out nodeInfoResponse
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.nodeInfoURL, nil)
	if err != nil {
		return out, fmt.Errorf("build node-info request: %w", err)
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return out, fmt.Errorf("read node-info: %w", err)
	}
	defer func() { _, _ = io.Copy(io.Discard, resp.Body); _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return out, fmt.Errorf("node-info returned %s", resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, nodeInfoMaxBytes))
	if err != nil {
		return out, fmt.Errorf("read node-info body: %w", err)
	}
	if err := json.Unmarshal(body, &out); err != nil {
		return out, fmt.Errorf("parse node-info: %w", err)
	}
	return out, nil
}

// NodeUUID returns the identity this collector reports under: node-info's
// hostUuid once it has answered, and the locally-derived fallback until then.
func (c *Collector) NodeUUID() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.nodeUUID
}
