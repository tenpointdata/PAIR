// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package poolwire is the single source of truth for the wire format
// nvpair-pool-manager speaks: the cluster-scoped HTTP surface peers use to
// report poolable capacity, and the JSON-RPC contract the broker relays.
//
// It exists for the reason nvpair-shared/noderec and nvpair-shared/schedulerwire
// exist: more than one process has to agree on these shapes — the pool manager
// on both ends of a peer call, the broker that relays them, and eventually the
// interface — and a wire format defined inside one of those processes is a wire
// format the others copy and then drift from.
//
// One decision here is worth stating plainly, because it changed during
// implementation and the reason generalizes. Pool capacity does NOT live on
// nvpair-node-info. That service is deliberately plaintext and unauthenticated —
// its own documentation calls that a considered trade for hardware inventory —
// and "this node has 40 GiB committed to a pool headed by that node" is cluster
// state, not hardware inventory. Publishing it unauthenticated would put the
// shape of a user's cluster on the LAN for anything that asks. So capacity is
// served from the pool manager's own cluster-mTLS port, readable only by pinned
// members, and node-info keeps reporting nothing but what the hardware says.
package poolwire

import "time"

// CapacityPath is the cluster-mTLS endpoint a peer GETs to learn what this node
// can contribute to a pool.
const CapacityPath = "/v1/pool/capacity"

// DeviceCapacity is one GPU's poolable memory, as its own node reports it.
//
// The four byte counts are kept separate rather than pre-subtracted into a
// single "free" number because they answer different questions and a user who
// cannot see them cannot act. Used is what is happening; Reserved is what the
// machine's owner declined to lend; Committed is what another pool already took.
// Collapsing them makes "why is my 24 GiB card offering 4 GiB" unanswerable.
type DeviceCapacity struct {
	// Index is the device's position in its node's GPU inventory. It is stable
	// for as long as that node's nvpair-node-info process is, which is the same
	// lifetime a lease can span, and it is what selects the device when a donor
	// process is started there.
	Index int `json:"index"`
	// Name is the adapter name, for display.
	Name string `json:"name,omitempty"`
	// TotalBytes is installed VRAM.
	TotalBytes uint64 `json:"totalBytes,omitempty"`
	// UsedBytes is what the driver reports in use, by anything at all — this
	// node's own engine, a game, another user's session.
	UsedBytes uint64 `json:"usedBytes,omitempty"`
	// ReservedBytes is memory the machine's owner has held back from pooling. A
	// desktop that also drives its own display should not lend its last
	// gigabyte, and only the person at that desk can say how much that is.
	ReservedBytes uint64 `json:"reservedBytes,omitempty"`
	// CommittedBytes is memory an existing pool lease already claimed. It is
	// tracked separately from UsedBytes because a lease is a promise made before
	// the memory is touched: between granting a lease and the model finishing
	// its load, the driver still reports the memory free, and a second pool
	// planned against it would overcommit the device.
	CommittedBytes uint64 `json:"committedBytes,omitempty"`
}

// FreeBytes is what a pool may actually plan against on this device.
//
// It saturates at zero rather than underflowing. These are unsigned, and the
// subtrahends come from three independent sources — a driver reading, a user
// preference, and a lease ledger — which can briefly sum to more than the
// device has. Wrapping that to sixteen exabytes would tell a planner the device
// is the largest in the cluster.
func (d DeviceCapacity) FreeBytes() uint64 {
	spoken := d.UsedBytes + d.ReservedBytes + d.CommittedBytes
	if spoken >= d.TotalBytes {
		return 0
	}
	return d.TotalBytes - spoken
}

// NodeCapacity is one node's complete answer about what it can contribute.
type NodeCapacity struct {
	// NodeUUID is the reporting node's stable host identity, so a consumer keys
	// capacity by the machine rather than by the address it was reached at.
	NodeUUID string `json:"nodeUuid"`
	// NodeName is the friendly name, for display.
	NodeName string `json:"nodeName,omitempty"`
	// DonorEnabled reports whether this node's owner has agreed to lend its
	// VRAM at all. It is false by default and it gates everything: lending a
	// machine's memory to a cluster is a decision its owner makes explicitly,
	// not a consequence of having paired.
	DonorEnabled bool `json:"donorEnabled"`
	// Devices is the per-GPU breakdown. Empty on a node with no GPU inventory,
	// which is a normal answer — a node can route without being able to serve.
	Devices []DeviceCapacity `json:"devices,omitempty"`
	// ObservedAtMs is when the reporting node took this reading, in Unix
	// milliseconds. A consumer compares it against its own clock only to judge
	// staleness in the large; the two clocks are not assumed to agree.
	ObservedAtMs int64 `json:"observedAtMs"`
	// TelemetryValid reports whether the underlying GPU reading was fresh. A
	// node whose telemetry is stale still answers, and says so, rather than
	// reporting zeros that would read as a full device.
	TelemetryValid bool `json:"telemetryValid"`
}

// ObservedAt renders ObservedAtMs as a time.
func (n NodeCapacity) ObservedAt() time.Time {
	return time.UnixMilli(n.ObservedAtMs)
}

// TotalFreeBytes sums what every device on this node could contribute. Zero
// when the node is not a donor at all, because capacity nobody has agreed to
// lend is not capacity.
func (n NodeCapacity) TotalFreeBytes() uint64 {
	if !n.DonorEnabled {
		return 0
	}
	var total uint64
	for _, d := range n.Devices {
		total += d.FreeBytes()
	}
	return total
}

// JSON-RPC methods the pool manager answers on its worker channel, relayed by
// the broker. Named with the same "area/verb" shape every other worker uses.
const (
	// MethodStatus returns this node's donor settings and its current view of
	// cluster capacity.
	MethodStatus = "pool/status"
	// MethodSetDonor enables or disables donation and sets the reservation.
	MethodSetDonor = "pool/donor/set"
	// MethodCapacity returns the cluster-wide capacity snapshot, refreshed.
	MethodCapacity = "pool/capacity"
	// MethodSetPeers tells the manager which peers to collect capacity from.
	// The pool manager does not browse mDNS itself: discovery is the scanner's
	// job and the broker already holds the consolidated directory, so pushing
	// the peer set down is one relay rather than a second browser competing for
	// the same multicast socket.
	MethodSetPeers = "pool/peers/set"
	// MethodPlan answers what a pool for a model would look like, without
	// forming one. It is the admission-control call: ask before downloading,
	// before starting anything, and before telling a user a model is available.
	MethodPlan = "pool/plan"
	// MethodForm brings a pool up.
	MethodForm = "pool/form"
	// MethodTeardown ends one.
	MethodTeardown = "pool/teardown"
	// NotifyUpdated is emitted when donor settings or the capacity view change.
	NotifyUpdated = "pool/updated"
	// NotifyPoolChanged is emitted when a pool forms or is torn down.
	NotifyPoolChanged = "pool/changed"
)

// Peer is one cluster member the manager should collect capacity from, as the
// broker knows it.
type Peer struct {
	NodeUUID string `json:"nodeUuid"`
	NodeName string `json:"nodeName,omitempty"`
	// Addresses are the peer's candidate host:port endpoints for the pool
	// manager's own port, best-ranked first. A list rather than one address
	// because a multi-homed peer has no single address every node can reach;
	// see nvpair-shared/reach, which walks exactly this shape.
	Addresses []string `json:"addresses"`
}

// DonorSettings is this node's own donation policy: the part of capacity that
// is a decision rather than a measurement.
type DonorSettings struct {
	// Enabled is the master switch. Default false.
	Enabled bool `json:"enabled"`
	// ReservedBytes is held back from every device. It is a per-device figure
	// rather than a node total, because it exists to protect the machine's
	// interactive use and each display runs on one card.
	ReservedBytes uint64 `json:"reservedBytes,omitempty"`
}

// Cluster-mTLS endpoints for donor leases and the tunnel that carries the ggml
// RPC stream. All of them live on the same port as CapacityPath and behind the
// same pinned-peer gate.
const (
	// LeasesPath grants a lease (POST). A lease is a donor node's promise to
	// hold memory for one pool.
	LeasesPath = "/v1/pool/leases"
	// LeasePathPrefix addresses one lease: POST renews it, DELETE releases it.
	LeasePathPrefix = LeasesPath + "/"
	// TunnelPathPrefix + leaseID is the HTTP/1.1 upgrade that becomes the ggml
	// RPC stream for that lease.
	TunnelPathPrefix = "/v1/pool/rpc/"
	// TunnelProtocol is the Upgrade token. It is not a registered protocol and
	// is not meant to be: it names this stream so an accidental request from
	// something else fails at the upgrade rather than being spliced into a
	// tensor backend.
	TunnelProtocol = "nvpair-ggml-rpc"
)

// LeaseRequest is a head node asking a donor to hold memory for one pool.
type LeaseRequest struct {
	// PoolID identifies the pool across its whole lifetime and across every
	// donor in it, so a donor can recognize a renewal and a head can release
	// every lease it holds after a restart.
	PoolID string `json:"poolId"`
	// ModelName is what the pool will serve. Carried for display and logs only —
	// a donor lends memory and never learns what is computed in it.
	ModelName string `json:"modelName,omitempty"`
	// DeviceIndexes are the donor devices the head plans to use, in its
	// inventory's own numbering.
	DeviceIndexes []int `json:"deviceIndexes"`
	// Bytes is what the head intends to place, across those devices together.
	Bytes uint64 `json:"bytes"`
	// TTLMs is how long the grant should stand without a renewal. It exists so a
	// head that dies does not strand a donor's memory forever: the lease lapses
	// and the memory returns on its own, with no cleanup message required from a
	// process that is gone.
	TTLMs int64 `json:"ttlMs,omitempty"`
}

// LeaseGrant is a donor's answer.
type LeaseGrant struct {
	LeaseID       string `json:"leaseId"`
	PoolID        string `json:"poolId"`
	DeviceIndexes []int  `json:"deviceIndexes"`
	Bytes         uint64 `json:"bytes"`
	ExpiresAtMs   int64  `json:"expiresAtMs"`
}

// LeaseTTLBounds constrain the requested TTL. A lease that lapses too eagerly
// turns a busy head's missed renewal into a dead pool; one that lapses too
// slowly strands a donor's memory after a head crashes.
const (
	DefaultLeaseTTL = 60 * time.Second
	MinLeaseTTL     = 10 * time.Second
	MaxLeaseTTL     = 10 * time.Minute
)

// TTL resolves the requested TTL against the bounds. An unset or out-of-range
// request takes the default rather than being refused: the head is asking for
// memory, and the renewal cadence is the donor's business to bound.
func (r LeaseRequest) TTL() time.Duration {
	if r.TTLMs <= 0 {
		return DefaultLeaseTTL
	}
	ttl := time.Duration(r.TTLMs) * time.Millisecond
	if ttl < MinLeaseTTL {
		return MinLeaseTTL
	}
	if ttl > MaxLeaseTTL {
		return MaxLeaseTTL
	}
	return ttl
}
