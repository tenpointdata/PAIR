// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"log/slog"
	"time"

	"nvpair-shared/applog"
	"nvpair-shared/poolwire"
)

// Version is stamped at build time via -ldflags "-X main.Version=...".
// See versions.json for the source of truth.
var Version = "dev"

// capacityCallTimeout bounds one cluster-wide capacity sweep initiated by a
// caller. It is longer than one peer call because a sweep is bounded by its
// slowest reachable peer, not by the sum of them.
const capacityCallTimeout = 10 * time.Second

// Manager is the worker's control surface: it answers the broker's JSON-RPC
// requests and announces changes.
//
// Frames are handled one at a time off the read loop, which is what lets the
// donor-settings read-modify-write below be a plain sequence rather than a
// transaction. Every other worker in the tree relies on the same property.
type Manager struct {
	codec     *Codec
	donor     *DonorState
	collector *Collector
	peers     *PeerCollector
	// leases is nil on a node with no donor command configured, which is also
	// how status reports that this node cannot lend memory at all — distinct
	// from a node that could but has declined.
	leases *LeaseStore
	// pools is the head side. Nil on a node with no pool-server command, which
	// is a separate capability from donating: a machine can lend its GPU without
	// being able to run a pool, and the reverse.
	pools *PoolManager

	cancel context.CancelFunc
}

func NewManager(codec *Codec, donor *DonorState, collector *Collector, peers *PeerCollector, leases *LeaseStore, pools *PoolManager) *Manager {
	return &Manager{codec: codec, donor: donor, collector: collector, peers: peers, leases: leases, pools: pools}
}

// Run announces readiness and serves the control channel until ctx is cancelled
// or the parent closes the pipe.
func (m *Manager) Run(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	m.cancel = cancel
	defer cancel()

	if err := m.codec.Notify("ready", map[string]string{"version": Version}); err != nil {
		return err
	}
	return m.readLoop(ctx)
}

func (m *Manager) readLoop(ctx context.Context) error {
	for {
		msg, err := m.codec.Read()
		if err != nil {
			if err == io.EOF || ctx.Err() != nil {
				return nil
			}
			log.Printf("JSON-RPC read error: %v", err)
			continue
		}
		m.handleMessage(ctx, msg)
		if ctx.Err() != nil {
			return nil
		}
	}
}

// donorParams is the params shape for poolwire.MethodSetDonor. Both fields are
// pointers so a caller can change one without restating the other — a user
// toggling donation off should not have to resend a reservation they set
// months ago, and a caller that omits it must not silently reset it to zero.
type donorParams struct {
	Enabled       *bool   `json:"enabled"`
	ReservedBytes *uint64 `json:"reservedBytes"`
}

// peersParams is the params shape for poolwire.MethodSetPeers.
type peersParams struct {
	Peers []poolwire.Peer `json:"peers"`
}

func (m *Manager) handleMessage(ctx context.Context, msg *Message) {
	if msg.Method == applog.SetLevelMethod {
		resolved, err := applog.HandleSetLevelParams(msg.Params)
		if msg.IsRequest() {
			if err != nil {
				m.codec.RespondError(msg.ID, -32602, err.Error())
				return
			}
			m.codec.Respond(msg.ID, map[string]string{"level": resolved})
		}
		if err != nil {
			slog.Warn("log/set-level rejected", "err", err)
		} else {
			slog.Info("log level changed", "level", resolved)
		}
		return
	}

	if !msg.IsRequest() {
		if msg.IsNotification() {
			slog.Debug("ignoring incoming notification", "method", msg.Method)
		}
		return
	}

	switch msg.Method {
	case poolwire.MethodStatus:
		m.codec.Respond(msg.ID, m.status(ctx, false))

	case poolwire.MethodCapacity:
		m.codec.Respond(msg.ID, m.status(ctx, true))

	case poolwire.MethodSetDonor:
		m.handleSetDonor(ctx, msg)

	case poolwire.MethodPlan:
		m.handlePlan(ctx, msg)

	case poolwire.MethodForm:
		m.handleForm(ctx, msg)

	case poolwire.MethodTeardown:
		m.handleTeardown(msg)

	case poolwire.MethodSetPeers:
		var p peersParams
		if err := json.Unmarshal(msg.Params, &p); err != nil {
			m.codec.RespondError(msg.ID, -32602, `invalid params: expected {"peers": [...]}`)
			return
		}
		m.peers.SetPeers(p.Peers)
		m.codec.Respond(msg.ID, map[string]any{"ok": true, "peers": len(p.Peers)})

	case "shutdown":
		m.codec.Respond(msg.ID, map[string]bool{"ok": true})
		if m.cancel != nil {
			m.cancel()
		}

	default:
		m.codec.RespondError(msg.ID, -32601, "method not found: "+msg.Method)
	}
}

// handleSetDonor applies a donation-policy change.
//
// The order is persist, then commit in memory, then announce. Persisting first
// means a save failure never leaves this node advertising a policy the disk does
// not agree with — which for this particular setting is the difference between
// "I lent my GPU until I said otherwise" and "I lent my GPU until the next
// restart, which re-enabled it".
func (m *Manager) handleSetDonor(ctx context.Context, msg *Message) {
	var p donorParams
	if err := json.Unmarshal(msg.Params, &p); err != nil {
		m.codec.RespondError(msg.ID, -32602, `invalid params: expected {"enabled": <bool>, "reservedBytes": <number>}`)
		return
	}
	if p.Enabled == nil && p.ReservedBytes == nil {
		m.codec.RespondError(msg.ID, -32602, "invalid params: nothing to change")
		return
	}

	next := m.donor.Settings()
	if p.Enabled != nil {
		next.Enabled = *p.Enabled
	}
	if p.ReservedBytes != nil {
		next.ReservedBytes = *p.ReservedBytes
	}

	changed, err := m.donor.Set(next)
	if err != nil {
		m.codec.RespondError(msg.ID, -32603, "failed to persist donor settings: "+err.Error())
		return
	}
	m.codec.Respond(msg.ID, map[string]any{"ok": true, "donor": next})

	if !changed {
		return
	}
	slog.Info("pool donor settings changed", "enabled", next.Enabled, "reservedBytes", next.ReservedBytes)
	// Announced after the response, so a caller's completion callback always
	// observes the notification strictly later than its own answer.
	if err := m.codec.Notify(poolwire.NotifyUpdated, m.status(ctx, false)); err != nil {
		slog.Warn("failed to announce pool update", "err", err)
	}
}

// statusResult is the shape both status and capacity answer with. One shape for
// both because the only difference between them is whether peers were re-asked,
// and a caller that has to branch on which call it made to read the answer is a
// caller that will get it wrong.
type statusResult struct {
	Donor poolwire.DonorSettings           `json:"donor"`
	Local poolwire.NodeCapacity            `json:"local"`
	Peers map[string]poolwire.NodeCapacity `json:"peers"`
	// Leases are the grants this node is currently honoring as a donor. Reported
	// because "my GPU is full and I did not start anything" is otherwise
	// unanswerable from the interface.
	Leases []poolwire.LeaseGrant `json:"leases"`
	// DonorReady reports whether this node has a backend command configured at
	// all. A node that has enabled donation but cannot run a backend is a
	// different problem from one that has declined, and collapsing the two hides
	// a misconfiguration behind a switch that looks correct.
	DonorReady bool `json:"donorReady"`
	// Pools are the pools this node currently heads.
	Pools []Pool `json:"pools"`
	// HeadReady reports whether this node can run a pool's server. Separate from
	// DonorReady because they are separate capabilities: a machine can lend its
	// GPU without being able to head a pool, and the reverse.
	HeadReady bool `json:"headReady"`
	// ClusterFreeBytes is what the whole cluster could contribute right now,
	// this node included. It is the headline number, and it is deliberately
	// computed here rather than by each consumer, because "sum the free bytes of
	// donors only" is a rule that has to be applied the same way everywhere.
	ClusterFreeBytes uint64 `json:"clusterFreeBytes"`
}

// status assembles the current view, optionally re-asking peers first.
func (m *Manager) status(ctx context.Context, refresh bool) statusResult {
	local := m.collector.Local(ctx)

	var peers map[string]poolwire.NodeCapacity
	if refresh {
		sweepCtx, cancel := context.WithTimeout(ctx, capacityCallTimeout)
		defer cancel()
		peers = m.peers.Collect(sweepCtx)
	} else {
		peers = m.peers.Last()
	}

	total := local.TotalFreeBytes()
	for _, capacity := range peers {
		total += capacity.TotalFreeBytes()
	}
	var leases []poolwire.LeaseGrant
	if m.leases != nil {
		leases = m.leases.Snapshot()
	}
	var pools []Pool
	headReady := false
	if m.pools != nil {
		pools = m.pools.Pools()
		headReady = m.pools.CanHead()
	}
	return statusResult{
		Donor:            m.donor.Settings(),
		Local:            local,
		Peers:            peers,
		Leases:           leases,
		DonorReady:       m.leases != nil,
		Pools:            pools,
		HeadReady:        headReady,
		ClusterFreeBytes: total,
	}
}

// planParams is the params shape for poolwire.MethodPlan and MethodForm.
type planParams struct {
	PoolID string `json:"poolId"`
	// ModelPath is a GGUF file on THIS node. Pooling reads a model's header to
	// plan against it, and it is the head that loads and distributes the
	// weights, so the file has to be here.
	ModelPath     string `json:"modelPath"`
	ContextTokens int    `json:"contextTokens"`
}

// teardownParams is the params shape for poolwire.MethodTeardown.
type teardownParams struct {
	PoolID string `json:"poolId"`
}

// defaultContextTokens is used when a caller does not say. A plan is only valid
// for the length it was built for, so this is a compromise in both directions:
// guessing large would refuse models that fit, and guessing small would form
// pools that run out of cache mid-conversation.
const defaultContextTokens = 4096

// handlePlan answers what a pool would look like, without forming one.
func (m *Manager) handlePlan(ctx context.Context, msg *Message) {
	if m.pools == nil {
		m.codec.RespondError(msg.ID, -32000, "pooling is not available on this node")
		return
	}
	var p planParams
	if err := json.Unmarshal(msg.Params, &p); err != nil || p.ModelPath == "" {
		m.codec.RespondError(msg.ID, -32602, `invalid params: expected {"modelPath": "<path to a .gguf>"}`)
		return
	}
	if p.ContextTokens <= 0 {
		p.ContextTokens = defaultContextTokens
	}
	result, err := m.pools.Plan(ctx, p.ModelPath, p.ContextTokens)
	if err != nil {
		m.codec.RespondError(msg.ID, -32000, err.Error())
		return
	}
	m.codec.Respond(msg.ID, result)
}

// handleForm brings a pool up.
func (m *Manager) handleForm(ctx context.Context, msg *Message) {
	if m.pools == nil {
		m.codec.RespondError(msg.ID, -32000, "pooling is not available on this node")
		return
	}
	var p planParams
	if err := json.Unmarshal(msg.Params, &p); err != nil || p.ModelPath == "" || p.PoolID == "" {
		m.codec.RespondError(msg.ID, -32602, `invalid params: expected {"poolId": "...", "modelPath": "<path to a .gguf>"}`)
		return
	}
	if p.ContextTokens <= 0 {
		p.ContextTokens = defaultContextTokens
	}

	pool, err := m.pools.Form(ctx, p.PoolID, p.ModelPath, p.ContextTokens)
	if err != nil {
		m.codec.RespondError(msg.ID, -32000, err.Error())
		return
	}
	m.codec.Respond(msg.ID, pool)
	if err := m.codec.Notify(poolwire.NotifyPoolChanged, m.status(ctx, false)); err != nil {
		slog.Warn("failed to announce pool formation", "err", err)
	}
}

// handleTeardown ends a pool.
func (m *Manager) handleTeardown(msg *Message) {
	if m.pools == nil {
		m.codec.RespondError(msg.ID, -32000, "pooling is not available on this node")
		return
	}
	var p teardownParams
	if err := json.Unmarshal(msg.Params, &p); err != nil || p.PoolID == "" {
		m.codec.RespondError(msg.ID, -32602, `invalid params: expected {"poolId": "..."}`)
		return
	}
	if err := m.pools.Teardown(p.PoolID); err != nil {
		m.codec.RespondError(msg.ID, -32000, err.Error())
		return
	}
	m.codec.Respond(msg.ID, map[string]bool{"ok": true})
	if err := m.codec.Notify(poolwire.NotifyPoolChanged, m.status(context.Background(), false)); err != nil {
		slog.Warn("failed to announce pool teardown", "err", err)
	}
}
