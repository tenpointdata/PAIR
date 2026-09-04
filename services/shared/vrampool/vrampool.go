// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package vrampool decides whether a model that fits on no single node can be
// held across several, and if so how to divide it.
//
// PAIR routes each request to one node, which means a model larger than the
// largest GPU in the cluster cannot run at all, however much memory the cluster
// has in total. Distributed inference closes that gap by putting consecutive
// layers on different machines — llama.cpp's ggml RPC backend does the actual
// sharding — but the decision of WHICH machines and HOW MANY layers each gets is
// not something llama.cpp can make. It cannot see the cluster, the other nodes'
// free memory, or what a link between them is worth. That decision is this
// package.
//
// Everything here is arithmetic over descriptions. No GPU, no engine, no
// network, no filesystem: a Device is a struct, a Model is a struct, and the
// planner is a function between them. That is deliberate and it is what makes
// the interesting part of distributed inference testable on a laptop with no
// second machine, which is the only way this logic gets exercised often enough
// to be trusted.
//
// Two properties of the underlying mechanism drive every rule below, and both
// are worth stating before the code assumes them:
//
//   - Layer-split inference is SEQUENTIAL. One device computes, hands its
//     hidden state to the next, and waits. More nodes buy capacity, never speed,
//     and every machine boundary adds a serialized round trip to every token. So
//     the planner uses the fewest, best-connected devices that fit rather than
//     spreading work evenly, which is the opposite of what a throughput
//     scheduler would do.
//   - A pool cannot survive losing a member. llama.cpp has no re-sharding and no
//     failover, so a donor that drops takes the pool and the in-flight request
//     with it. That makes link quality a correctness input rather than a
//     performance one, and it is why a donor is GATED on its link rather than
//     merely ranked by it.
package vrampool

import (
	"fmt"

	"nvpair-shared/linkq"
)

// Device is one GPU offered to a pool, described from the prospective head
// node's point of view — Link and Stability are properties of the path from the
// head to this device's node, not of the device.
type Device struct {
	// NodeUUID identifies the machine holding this GPU.
	NodeUUID string
	// NodeName is for display and log lines.
	NodeName string
	// Index is the GPU's index on its own node, which is what selects it when
	// the donor process is started there.
	Index int
	// Name is the adapter name, for display.
	Name string
	// TotalBytes is the device's installed VRAM.
	TotalBytes uint64
	// FreeBytes is what is available to a pool right now: installed memory less
	// what is in use, less whatever the node's owner reserved for themselves,
	// less what another pool has already committed. The node computes it, because
	// only the node knows the last two.
	FreeBytes uint64
	// Link is the path class from the head to this device's node.
	Link linkq.Class
	// Stability is the path's reachability score in [0,1]. A donor that comes and
	// goes is worse than a donor with less memory, because losing one is not a
	// slow request, it is a failed pool.
	Stability float64
	// MediaKnown reports whether both ends of the path were positively
	// identified. "Not known to be wireless" is not the same claim as "known to
	// be wired", and a strict policy insists on the second one.
	MediaKnown bool
}

// String renders a device for a log line or a rejection message.
func (d Device) String() string {
	return fmt.Sprintf("%s/gpu%d (%s, %.1f GiB free, %s)",
		d.NodeName, d.Index, d.Name, gib(d.FreeBytes), d.Link)
}

func gib(b uint64) float64 { return float64(b) / (1 << 30) }

// Model is what has to fit, in the terms GGUF metadata already provides. Every
// field is readable from a model file's header without loading a single tensor,
// which is what lets admission control answer before anything is downloaded or
// started.
type Model struct {
	// Name is the model identifier, for display and for the plan's identity.
	Name string
	// WeightBytes is the total size of the model's tensors.
	WeightBytes uint64
	// Layers is the repeating block count.
	Layers int
	// NonRepeatingBytes is the token-embedding and output tensors — everything
	// that is not one of the Layers blocks. It is charged to the head node,
	// where llama.cpp keeps it.
	NonRepeatingBytes uint64
	// KVBytesPerLayerPerToken is the key-value cache cost of one token on one
	// layer: 2 (keys and values) times the KV width times the element size. For
	// a grouped-query model the KV width is the grouped width, not the model's
	// embedding width, which is the difference between a plan that fits and one
	// that overcommits by a factor of eight.
	KVBytesPerLayerPerToken uint64
}

// KVBytesPerLayerPerToken computes the field of the same name from a model's
// grouped KV width and element size, so a caller reading GGUF metadata does not
// have to remember the factor of two.
func KVBytesPerLayerPerToken(kvWidth, elementBytes uint64) uint64 {
	return 2 * kvWidth * elementBytes
}

// PerLayerWeightBytes is the average size of one repeating block.
//
// It is an average, and layers genuinely differ — the first and last blocks of
// some architectures are not the same size as the middle ones. The error is a
// few percent and it is absorbed by the policy's headroom, which exists for
// exactly this class of approximation. Reading every tensor's size from the
// header instead would be more precise and would make admission control depend
// on having the file, which is the thing it is supposed to answer before.
func (m Model) PerLayerWeightBytes() uint64 {
	if m.Layers <= 0 {
		return 0
	}
	repeating := m.WeightBytes
	if m.NonRepeatingBytes < repeating {
		repeating -= m.NonRepeatingBytes
	}
	return repeating / uint64(m.Layers)
}

// PerLayerBytes is what one layer costs on a device: its weights plus the
// key-value cache for a full context.
//
// KV is charged per layer rather than to the head because a layer's cache lives
// with the layer. This is the term that makes a long context expensive on a
// pool: doubling the context does not touch the weights and doubles the cache,
// and it is the cache that decides whether the last device has room.
func (m Model) PerLayerBytes(contextTokens int) uint64 {
	if contextTokens < 0 {
		contextTokens = 0
	}
	return m.PerLayerWeightBytes() + m.KVBytesPerLayerPerToken*uint64(contextTokens)
}

// TotalBytes is what the whole model costs at this context length, ignoring
// per-device overheads. It is the number to compare against the cluster's total
// free VRAM when answering "could this ever fit".
func (m Model) TotalBytes(contextTokens int) uint64 {
	return m.NonRepeatingBytes + m.PerLayerBytes(contextTokens)*uint64(max(m.Layers, 0))
}

// Validate reports whether a model description can be planned against at all.
//
// Exported because the description usually comes from outside this package — a
// GGUF header, a manifest, a caller's JSON — and the useful moment to reject a
// malformed one is when it is read, not when a plan built from it silently
// overcommits every device in the cluster.
func (m Model) Validate() error {
	switch {
	case m.Layers <= 0:
		return fmt.Errorf("vrampool: model %q has no layer count", m.Name)
	case m.WeightBytes == 0:
		return fmt.Errorf("vrampool: model %q has no weight size", m.Name)
	case m.NonRepeatingBytes > m.WeightBytes:
		return fmt.Errorf("vrampool: model %q claims more non-repeating bytes than total weights", m.Name)
	}
	return nil
}

// Policy is the operator-facing part of planning: how much of a device's free
// memory a pool may claim, and which donors are acceptable at all.
//
// It is a struct rather than a set of constants because the right answers differ
// by cluster — a rack of wired workstations and a house with one laptop on Wi-Fi
// want different rules — and because a policy that can be constructed in a test
// is a policy whose edges get tested.
type Policy struct {
	// Headroom is the fraction of a device's free VRAM a plan may use, in (0,1].
	// Zero means DefaultHeadroom.
	//
	// Planning to the last free byte fails on allocator fragmentation, on the
	// compute buffer growing with batch size, and on the first request whose
	// context runs longer than the one that was planned for. The margin looks
	// wasteful written down and is the difference between a pool that loads and
	// one that dies at ninety-six percent.
	Headroom float64
	// PerDeviceOverheadBytes is charged to every participating device for the
	// compute buffer and allocator slack. Zero means DefaultPerDeviceOverhead.
	// It is separate from Headroom because it is roughly fixed per device rather
	// than proportional to the device's size, so a small GPU pays it too.
	PerDeviceOverheadBytes uint64
	// MaxDonors caps how many machines beyond the head may join. Zero means
	// DefaultMaxDonors. Every donor is another serialized hop per token and
	// another single point of failure, so the cap is low on purpose.
	MaxDonors int
	// AllowWiFiDonors permits a donor reached over ClassWiFi. Off by default:
	// generation across a wireless hop is slow and, worse, a wireless hop is the
	// one most likely to drop and take the pool with it.
	AllowWiFiDonors bool
	// AllowUnknownLink permits a donor whose link has not been classified.
	// Off by default, because ClassUnknown means no evidence, and admitting a
	// donor on no evidence is how a pool acquires the member that kills it.
	AllowUnknownLink bool
	// RequireKnownMedia insists that both ends of a donor's path be positively
	// identified before it may join.
	RequireKnownMedia bool
	// MinStability is the reachability score a donor must meet, in [0,1]. Zero
	// means DefaultMinStability.
	MinStability float64
}

// Policy defaults. Each is conservative in the same direction: a pool that
// declines to form costs the user an error message, and a pool that forms and
// then dies costs them the request, the load time, and their confidence.
const (
	// DefaultHeadroom leaves a tenth of each device free.
	DefaultHeadroom = 0.90
	// DefaultPerDeviceOverhead covers llama.cpp's compute buffer and allocator
	// slack at ordinary batch sizes.
	DefaultPerDeviceOverhead uint64 = 512 << 20
	// DefaultMaxDonors allows a head plus three donors. Beyond that the added
	// per-token latency usually costs more than the added capacity is worth.
	DefaultMaxDonors = 3
	// DefaultMinStability demands a nearly unbroken reachability record.
	DefaultMinStability = 0.95
)

// withDefaults returns p with zero fields filled in.
func (p Policy) withDefaults() Policy {
	if p.Headroom <= 0 || p.Headroom > 1 {
		p.Headroom = DefaultHeadroom
	}
	if p.PerDeviceOverheadBytes == 0 {
		p.PerDeviceOverheadBytes = DefaultPerDeviceOverhead
	}
	if p.MaxDonors <= 0 {
		p.MaxDonors = DefaultMaxDonors
	}
	if p.MinStability <= 0 {
		p.MinStability = DefaultMinStability
	}
	return p
}

// usable is how much of a device this policy will plan against: its free memory
// scaled by headroom, less the per-device overhead. It saturates at zero rather
// than underflowing, which matters because these are unsigned and a small device
// can owe more overhead than it has memory.
func (p Policy) usable(d Device) uint64 {
	scaled := uint64(float64(d.FreeBytes) * p.Headroom)
	if scaled <= p.PerDeviceOverheadBytes {
		return 0
	}
	return scaled - p.PerDeviceOverheadBytes
}
