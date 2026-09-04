// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"net"
	"slices"
	"strconv"
	"time"

	"nvpair-shared/noderec"
	"nvpair-shared/poolwire"
)

// poolManagerHTTPPort is the fixed LAN port the broker tells nvpair-pool-manager
// to serve its cluster-scoped surface on, and the port it registers as the pl
// service so peers can read this node's poolable capacity. Next free after
// engine-manager's ec in the 143xx range.
const poolManagerHTTPPort = 14324

// poolPeerSyncInterval is how often the broker re-derives the pool manager's
// peer set from the discovery directory.
//
// A timer rather than a subscription to directory events, because the set the
// pool manager needs is a filtered projection — trusted members that advertise
// the pool service — and recomputing it costs a map walk over a handful of
// nodes. Pushing only on change keeps the worker from re-collecting capacity
// because an unrelated node's GPU utilization moved.
const poolPeerSyncInterval = 5 * time.Second

// poolMethods are the client-facing methods the broker relays straight to the
// pool manager. Listed rather than prefix-matched so an unrecognized pool/*
// method is a method-not-found from the broker instead of an unexplained silence
// from a worker that may not be running.
var poolMethods = []string{
	poolwire.MethodStatus,
	poolwire.MethodCapacity,
	poolwire.MethodSetDonor,
	poolwire.MethodPlan,
	poolwire.MethodForm,
	poolwire.MethodTeardown,
}

// isPoolMethod reports whether a request belongs to the pool manager.
func isPoolMethod(method string) bool { return slices.Contains(poolMethods, method) }

// spawnPoolManager starts nvpair-pool-manager and registers its cluster port.
func (b *Broker) spawnPoolManager() (supervisedHandle, error) {
	args := append(b.logLevelArgs(), b.clusterDirArgs()...)
	args = append(args, "--port", strconv.Itoa(poolManagerHTTPPort))
	if b.poolDonorCommand != "" {
		args = append(args, "--donor-command", b.poolDonorCommand)
	}
	if b.poolHeadCommand != "" {
		args = append(args, "--head-command", b.poolHeadCommand)
	}
	if b.poolAllowWiFiDonors {
		args = append(args, "--allow-wifi-donors")
	}

	w, err := startRPCWorker("pool-manager", b.poolManagerPath, args, b.forwardPoolManagerNotification)
	if err != nil {
		return nil, err
	}
	b.setPoolManager(w)
	b.registerService(noderec.RegisterParams{Service: noderec.ServicePool, Port: poolManagerHTTPPort})
	slog.Info("pool-manager started", "path", b.poolManagerPath, "pid", w.cmd.Process.Pid,
		"port", poolManagerHTTPPort, "canDonate", b.poolDonorCommand != "", "canHead", b.poolHeadCommand != "")
	return w, nil
}

// clearPoolManagerState is the pool manager's supervisor clearHandle: on a crash
// the handle is dropped and the service is unregistered, so peers stop being
// told this node serves a port nothing is listening on. A restart re-registers.
func (b *Broker) clearPoolManagerState() {
	b.setPoolManager(nil)
	b.unregisterService(noderec.ServicePool)
}

// forwardPoolManagerNotification relays the worker's notifications to clients.
func (b *Broker) forwardPoolManagerNotification(method string, params json.RawMessage) {
	if b.dispatchErrorsNotif("pool-manager", method, params) {
		return
	}
	switch method {
	case "ready":
		slog.Info("pool-manager reported ready", "params", string(params))
	case poolwire.NotifyUpdated, poolwire.NotifyPoolChanged:
		if err := b.codec.Notify(method, params); err != nil {
			slog.Warn("failed to relay a pool notification", "method", method, "err", err)
		}
	default:
		slog.Debug("ignoring pool-manager notification", "method", method)
	}
}

// relayToPoolManager forwards a pool request to the worker and relays its
// answer.
func (b *Broker) relayToPoolManager(msg *Message) {
	pm := b.getPoolManager()
	if pm == nil {
		if err := b.codec.RespondError(msg.ID, -32000, "pool-manager not available"); err != nil {
			slog.Warn("failed to reject a pool request", "method", msg.Method, "err", err)
		}
		return
	}
	id := msg.ID
	method := msg.Method
	// The async, unbounded relay rather than the bounded Call: forming a pool
	// leases donors, starts backends on other machines, and waits for a model to
	// load across the network. That is minutes of legitimate work, and a request
	// timeout here would abandon a pool that was still coming up.
	relayErr := pm.RelayRequest(method, msg.Params, func(result json.RawMessage, rpcErr *RPCError, err error) {
		switch {
		case err != nil:
			if e := b.codec.RespondError(id, -32000, "pool-manager call failed: "+err.Error()); e != nil {
				slog.Warn("failed to relay a pool error", "method", method, "err", e)
			}
		case rpcErr != nil:
			if e := b.codec.RespondError(id, rpcErr.Code, rpcErr.Message); e != nil {
				slog.Warn("failed to relay a pool error", "method", method, "err", e)
			}
		default:
			if e := b.codec.Respond(id, result); e != nil {
				slog.Warn("failed to relay a pool result", "method", method, "err", e)
			}
		}
	})
	if relayErr != nil {
		if err := b.codec.RespondError(msg.ID, -32000, "pool-manager call failed: "+relayErr.Error()); err != nil {
			slog.Warn("failed to reject a pool request", "method", msg.Method, "err", err)
		}
	}
}

// runPoolPeerSync keeps the pool manager's peer set current for as long as the
// broker runs.
//
// The pool manager does not browse mDNS. Discovery is the scanner's job and the
// broker already holds the consolidated directory, so a second browser in the
// worker would compete for the same multicast socket to learn what the parent
// already knows.
func (b *Broker) runPoolPeerSync(ctx context.Context) {
	ticker := time.NewTicker(poolPeerSyncInterval)
	defer ticker.Stop()

	var last []poolwire.Peer
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			peers := b.poolPeers()
			if samePoolPeers(last, peers) {
				continue
			}
			pm := b.getPoolManager()
			if pm == nil {
				continue
			}
			if err := pm.Notify(poolwire.MethodSetPeers, map[string]any{"peers": peers}); err != nil {
				slog.Debug("failed to push the pool peer set", "err", err)
				continue
			}
			last = peers
			slog.Debug("pushed the pool peer set", "peers", len(peers))
		}
	}
}

// poolPeers projects the discovery directory onto the pool manager's peer shape.
//
// Only TRUSTED nodes are included. An untrusted node advertising the pool
// service is one this node holds no pin for, so every call to it would fail the
// client-side gate before a connection was opened — listing it would cost a
// pointless probe per sweep and would show a user a peer that can never
// contribute.
func (b *Broker) poolPeers() []poolwire.Peer {
	nodes := b.relayDir.Snapshot(noderec.ServicePool)
	out := make([]poolwire.Peer, 0, len(nodes))
	for _, n := range nodes {
		if !n.Trusted || n.HostUUID == b.nodeID {
			continue
		}
		status, ok := n.Services[noderec.ServicePool]
		if !ok || status.Port <= 0 {
			continue
		}
		addrs := poolAddresses(n, status.Port)
		if len(addrs) == 0 {
			continue
		}
		out = append(out, poolwire.Peer{NodeUUID: n.HostUUID, NodeName: n.Name, Addresses: addrs})
	}
	return out
}

// poolAddresses renders a node's published addresses as host:port candidates,
// in the node's own ranked order.
//
// The full list, not just the canonical address: a multi-homed node has no
// single address every peer can reach, and the pool manager walks these through
// nvpair-shared/reach exactly as every other dialer in the tree does.
func poolAddresses(n noderec.DirectoryNode, port int) []string {
	seen := make(map[string]struct{}, len(n.IPs)+1)
	out := make([]string, 0, len(n.IPs)+1)
	for _, ip := range append([]string{n.IP}, n.IPs...) {
		if ip == "" {
			continue
		}
		if _, dup := seen[ip]; dup {
			continue
		}
		seen[ip] = struct{}{}
		out = append(out, net.JoinHostPort(ip, strconv.Itoa(port)))
	}
	return out
}

// samePoolPeers reports whether two peer sets are equivalent, so an unchanged
// directory produces no push.
func samePoolPeers(a, b []poolwire.Peer) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].NodeUUID != b[i].NodeUUID || a[i].NodeName != b[i].NodeName {
			return false
		}
		if !slices.Equal(a[i].Addresses, b[i].Addresses) {
			return false
		}
	}
	return true
}
