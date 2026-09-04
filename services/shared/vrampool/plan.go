// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package vrampool

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"nvpair-shared/linkq"
)

// Assignment is one device's share of a pooled model.
type Assignment struct {
	// Device is the GPU this share lands on.
	Device Device
	// FirstLayer is the index of this device's first repeating block, and Layers
	// is how many consecutive blocks it holds. Consecutive on purpose: layers are
	// evaluated in order, so splitting a device's share into non-adjacent ranges
	// would cross the network twice for no benefit.
	FirstLayer int
	Layers     int
	// WeightBytes, KVBytes, and OverheadBytes are what this share costs, split
	// out so a user asking why a device is full gets an answer rather than a
	// total.
	WeightBytes   uint64
	KVBytes       uint64
	OverheadBytes uint64
	// TotalBytes is the sum, and what is compared against the device's usable
	// memory.
	TotalBytes uint64
	// IsHead marks the device running llama-server, which also carries the
	// model's non-repeating tensors.
	IsHead bool
}

// Plan is a formable pool: who participates, what each holds, and what it will
// cost to run.
type Plan struct {
	// Model and ContextTokens are the requirement this plan satisfies. A plan is
	// only valid for the context length it was built for, because the key-value
	// cache is most of what makes a long context not fit.
	Model         Model
	ContextTokens int
	// Assignments are in layer order, head first.
	Assignments []Assignment
	// TensorSplit is the per-device proportion llama.cpp is given, in the same
	// order as Assignments.
	TensorSplit []float64
	// RequiredBytes is the plan's total VRAM cost across every device.
	RequiredBytes uint64
	// Boundaries is how many times the layer sequence crosses from one machine
	// to another. It is the plan's latency term: each boundary adds a serialized
	// round trip to every generated token, so a plan with one boundary and a
	// plan with three are very different things at the same capacity.
	Boundaries int
	// SlowestLink is the worst path class among participating devices, which is
	// the one that will decide how the pool actually feels.
	SlowestLink linkq.Class
}

// Nodes returns the distinct machines the plan uses, head first, in layer order.
func (p *Plan) Nodes() []string {
	seen := make(map[string]struct{}, len(p.Assignments))
	var out []string
	for _, a := range p.Assignments {
		if _, dup := seen[a.Device.NodeUUID]; dup {
			continue
		}
		seen[a.Device.NodeUUID] = struct{}{}
		out = append(out, a.Device.NodeUUID)
	}
	return out
}

// TensorSplitArg renders the split as llama.cpp's --tensor-split expects it.
// Proportions rather than layer counts, because that is what the flag takes, and
// three decimal places because the loader rounds to layer boundaries anyway and
// more digits only make the command line harder to read in a log.
func (p *Plan) TensorSplitArg() string {
	parts := make([]string, len(p.TensorSplit))
	for i, f := range p.TensorSplit {
		parts[i] = strconv.FormatFloat(f, 'f', 3, 64)
	}
	return strings.Join(parts, ",")
}

// Rejection records one device the planner would not use, and why. It is
// returned even on success: "the cluster has 60 GiB free and my model needs 40,
// so why did this fail" is the first question a pool that declines to form
// provokes, and the answer is almost always here.
type Rejection struct {
	Device Device
	Reason string
}

// Outcome is the planner's full answer: a plan when one exists, the devices that
// were turned away, and how far short the rest fell.
type Outcome struct {
	// Plan is nil when the model cannot be held by the eligible devices.
	Plan *Plan
	// Rejected lists devices excluded by policy, with reasons.
	Rejected []Rejection
	// UnplacedLayers and ShortfallBytes describe the gap when Plan is nil: how
	// many blocks had nowhere to go and roughly how much more VRAM would have
	// been needed. Reported so an interface can say "about 12 GiB short" rather
	// than "it does not fit".
	UnplacedLayers int
	ShortfallBytes uint64
}

// Fits reports whether the outcome produced a usable plan.
func (o Outcome) Fits() bool { return o.Plan != nil }

// Build plans a pool for one model at one context length.
//
// headNodeUUID is the machine that will run llama-server. It is an input rather
// than something chosen here because the head is not a free choice: it is the
// node the request arrived on, and moving the request to a different node is
// ordinary routing, which PAIR already does and does not need a pool for.
//
// An error means the inputs cannot be planned against at all — a model with no
// layer count, a nonsensical context. A model that simply does not fit is not an
// error; it is an Outcome with no Plan, which is a routine and answerable
// result.
func Build(model Model, contextTokens int, devices []Device, headNodeUUID string, policy Policy) (Outcome, error) {
	if err := model.Validate(); err != nil {
		return Outcome{}, err
	}
	if contextTokens <= 0 {
		return Outcome{}, fmt.Errorf("vrampool: context length must be positive, got %d", contextTokens)
	}
	if headNodeUUID == "" {
		return Outcome{}, fmt.Errorf("vrampool: no head node named")
	}
	policy = policy.withDefaults()

	perLayer := model.PerLayerBytes(contextTokens)
	if perLayer == 0 {
		return Outcome{}, fmt.Errorf("vrampool: model %q has zero-sized layers", model.Name)
	}

	head, donors, rejected := partition(devices, headNodeUUID, policy)
	if len(head) == 0 {
		return Outcome{
			Rejected:       rejected,
			UnplacedLayers: model.Layers,
			ShortfallBytes: model.TotalBytes(contextTokens),
		}, nil
	}

	ordered := append(head, donors...)
	assignments, remaining := allocate(ordered, model, contextTokens, perLayer, policy)
	if remaining > 0 {
		shortfall := uint64(remaining) * perLayer
		if len(assignments) == 0 {
			// Nothing was placed at all, which means the head could not even
			// hold the non-repeating tensors. Those are part of the gap too, and
			// omitting them would understate it by the size of an embedding
			// table.
			shortfall += model.NonRepeatingBytes
		}
		return Outcome{
			Rejected:       rejected,
			UnplacedLayers: remaining,
			ShortfallBytes: shortfall,
		}, nil
	}

	return Outcome{Plan: finish(model, contextTokens, assignments), Rejected: rejected}, nil
}

// partition splits devices into the head node's own and the donors that pass
// policy, recording why each excluded device was excluded.
//
// The head node's devices are never gated. Whatever their link class says, they
// are on the machine running the server, and refusing them would refuse the pool
// its own memory over a measurement about a path that does not exist.
func partition(devices []Device, headNodeUUID string, policy Policy) (head, donors []Device, rejected []Rejection) {
	for _, d := range devices {
		if d.NodeUUID == headNodeUUID {
			if policy.usable(d) == 0 {
				rejected = append(rejected, Rejection{d, "no usable VRAM after headroom and compute-buffer overhead"})
				continue
			}
			head = append(head, d)
			continue
		}
		if reason, ok := donorReason(d, policy); !ok {
			rejected = append(rejected, Rejection{d, reason})
			continue
		}
		donors = append(donors, d)
	}
	sort.SliceStable(head, func(i, j int) bool { return betterDevice(head[i], head[j]) })
	sort.SliceStable(donors, func(i, j int) bool { return betterDevice(donors[i], donors[j]) })
	return head, donors, rejected
}

// donorReason reports whether a device may join as a donor, and why not when it
// may not.
//
// Every gate here is about the pool surviving rather than the pool being fast.
// That is the distinction worth keeping in mind while reading them: a slow donor
// makes a slow pool, which is a disappointment, and an unreliable donor makes a
// pool that dies mid-request, which is a failure.
func donorReason(d Device, policy Policy) (string, bool) {
	switch d.Link {
	case linkq.ClassWAN:
		return "link is wide-area; a pool cannot survive the latency or the drops", false
	case linkq.ClassUnknown:
		if !policy.AllowUnknownLink {
			return "link has not been measured", false
		}
	case linkq.ClassWiFi:
		if !policy.AllowWiFiDonors {
			return "link is wireless; enable wireless donors to use it anyway", false
		}
	}
	if policy.RequireKnownMedia && !d.MediaKnown {
		return "link media could not be positively identified", false
	}
	if d.Stability < policy.MinStability {
		return fmt.Sprintf("link stability %.2f is below the %.2f a pool member needs", d.Stability, policy.MinStability), false
	}
	if policy.usable(d) == 0 {
		return "no usable VRAM after headroom and compute-buffer overhead", false
	}
	return "", true
}

// betterDevice orders devices for greedy allocation: best link first, then
// steadiest, then largest, then by identity so the answer is reproducible.
//
// Link before size is the whole ordering decision. Filling a big wireless GPU
// before a smaller wired one produces a pool that holds the model and generates
// tokens at the speed of the radio, which is the trade nobody wants and nobody
// asked for.
func betterDevice(a, b Device) bool {
	if la, lb := linkRank(a.Link), linkRank(b.Link); la != lb {
		return la < lb
	}
	if a.Stability != b.Stability {
		return a.Stability > b.Stability
	}
	if a.FreeBytes != b.FreeBytes {
		return a.FreeBytes > b.FreeBytes
	}
	if a.NodeUUID != b.NodeUUID {
		return a.NodeUUID < b.NodeUUID
	}
	return a.Index < b.Index
}

// linkRank orders link classes best to worst. ClassUnknown sorts last rather
// than first: the constant's zero value must not read as the best possible link.
func linkRank(c linkq.Class) int {
	switch c {
	case linkq.ClassLocal:
		return 0
	case linkq.ClassLAN:
		return 1
	case linkq.ClassWiFi:
		return 2
	case linkq.ClassWAN:
		return 3
	default:
		return 4
	}
}

// allocate fills devices in order until every layer is placed, returning the
// assignments and how many layers had nowhere to go.
//
// Greedy fill rather than a proportional spread, and this is the single most
// consequential choice in the package. Spreading layers evenly across every
// eligible device maximizes the number of machine boundaries, and each boundary
// is a serialized round trip on every token — so an even split of a model that
// would have fitted on two machines across five is several times slower for no
// gain at all. Filling the best device first uses the fewest machines that can
// hold the model, which is also the fastest arrangement that can.
func allocate(ordered []Device, model Model, contextTokens int, perLayer uint64, policy Policy) ([]Assignment, int) {
	remaining := model.Layers
	next := 0
	donorNodes := make(map[string]struct{})
	var out []Assignment

	for _, d := range ordered {
		if remaining == 0 {
			break
		}
		isHead := len(out) == 0
		if !isHead {
			if _, known := donorNodes[d.NodeUUID]; !known {
				if len(donorNodes) >= policy.MaxDonors {
					// Another machine would exceed the donor cap. Additional
					// GPUs on machines already in the pool are still fair game,
					// which is why this checks nodes rather than devices.
					continue
				}
			}
		}

		capacity := policy.usable(d)
		if isHead {
			// The head also carries the embedding and output tensors, which
			// llama.cpp keeps on the machine running the server.
			if capacity <= model.NonRepeatingBytes {
				// The head cannot even hold the non-repeating tensors, so no
				// arrangement of donors helps: there is nowhere to put them.
				return out, remaining
			}
			capacity -= model.NonRepeatingBytes
		}

		layers := int(capacity / perLayer)
		if layers > remaining {
			layers = remaining
		}
		if layers == 0 && !isHead {
			continue
		}

		a := Assignment{
			Device:        d,
			FirstLayer:    next,
			Layers:        layers,
			WeightBytes:   model.PerLayerWeightBytes() * uint64(layers),
			KVBytes:       model.KVBytesPerLayerPerToken * uint64(contextTokens) * uint64(layers),
			OverheadBytes: policy.PerDeviceOverheadBytes,
			IsHead:        isHead,
		}
		if isHead {
			a.WeightBytes += model.NonRepeatingBytes
		}
		a.TotalBytes = a.WeightBytes + a.KVBytes + a.OverheadBytes
		out = append(out, a)

		if !isHead {
			donorNodes[d.NodeUUID] = struct{}{}
		}
		next += layers
		remaining -= layers
	}
	return out, remaining
}

// finish computes the derived fields of a complete allocation.
func finish(model Model, contextTokens int, assignments []Assignment) *Plan {
	p := &Plan{
		Model:         model,
		ContextTokens: contextTokens,
		Assignments:   assignments,
		TensorSplit:   make([]float64, len(assignments)),
		SlowestLink:   linkq.ClassLocal,
	}
	total := float64(model.Layers)
	lastNode := ""
	for i, a := range assignments {
		p.TensorSplit[i] = float64(a.Layers) / total
		p.RequiredBytes += a.TotalBytes
		if !a.IsHead && linkRank(a.Device.Link) > linkRank(p.SlowestLink) {
			p.SlowestLink = a.Device.Link
		}
		if lastNode != "" && a.Device.NodeUUID != lastNode {
			p.Boundaries++
		}
		lastNode = a.Device.NodeUUID
	}
	return p
}

// FitsOnOne reports the single best device that could hold the whole model,
// which is the question to ask BEFORE planning a pool.
//
// A model that fits on one node should be routed there, not pooled: routing is
// faster, survives that node going away, and is what PAIR already does well.
// Pooling is for the case where the alternative is not running at all.
func FitsOnOne(model Model, contextTokens int, devices []Device, policy Policy) (Device, bool) {
	if model.Validate() != nil || contextTokens <= 0 {
		return Device{}, false
	}
	policy = policy.withDefaults()
	need := model.TotalBytes(contextTokens)

	best := Device{}
	found := false
	for _, d := range devices {
		if policy.usable(d) < need {
			continue
		}
		if !found || betterDevice(d, best) {
			best = d
			found = true
		}
	}
	return best, found
}
