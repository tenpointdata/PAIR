// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package vrampool

import (
	"strings"
	"testing"

	"nvpair-shared/linkq"
)

const (
	gb   = uint64(1) << 30
	head = "head-node"
)

// wired builds a healthy LAN donor, which is what most tests want; the
// interesting cases then change one field.
func wired(node string, index int, free uint64) Device {
	return Device{
		NodeUUID:   node,
		NodeName:   node,
		Index:      index,
		Name:       "Test GPU",
		TotalBytes: free,
		FreeBytes:  free,
		Link:       linkq.ClassLAN,
		Stability:  1,
		MediaKnown: true,
	}
}

func headDevice(free uint64) Device {
	d := wired(head, 0, free)
	d.Link = linkq.ClassLocal
	return d
}

// A 32-layer model with 1 GiB layers and a small non-repeating share, which
// makes every capacity assertion below readable in whole gigabytes.
func testModel() Model {
	return Model{
		Name:                    "test-32b",
		Layers:                  32,
		NonRepeatingBytes:       2 * gb,
		WeightBytes:             34 * gb, // 2 GiB non-repeating plus 32 x 1 GiB
		KVBytesPerLayerPerToken: 0,       // charged separately where a test needs it
	}
}

func TestBuildRejectsUnusableInputs(t *testing.T) {
	m := testModel()
	devices := []Device{headDevice(80 * gb)}

	if _, err := Build(Model{Name: "no-layers", WeightBytes: gb}, 4096, devices, head, Policy{}); err == nil {
		t.Error("a model with no layer count should be an error, not a non-fit")
	}
	if _, err := Build(Model{Name: "no-weights", Layers: 32}, 4096, devices, head, Policy{}); err == nil {
		t.Error("a model with no weight size should be an error")
	}
	if _, err := Build(m, 0, devices, head, Policy{}); err == nil {
		t.Error("a zero context should be an error")
	}
	if _, err := Build(m, 4096, devices, "", Policy{}); err == nil {
		t.Error("planning with no head node should be an error")
	}
}

// A model that fits on one device should be routed, not pooled, and the planner
// says so before anything else is decided.
func TestFitsOnOnePrefersTheBestDevice(t *testing.T) {
	m := testModel()
	devices := []Device{
		wired("peer-wifi", 0, 200*gb),
		wired("peer-lan", 0, 60*gb),
		headDevice(60 * gb),
	}
	devices[0].Link = linkq.ClassWiFi

	got, ok := FitsOnOne(m, 4096, devices, Policy{})
	if !ok {
		t.Fatal("a 34 GiB model should fit on a 60 GiB device")
	}
	if got.NodeUUID != head {
		t.Fatalf("chose %s; the local device should win over a larger wireless one", got.NodeUUID)
	}
}

func TestFitsOnOneRefusesWhenNothingIsBigEnough(t *testing.T) {
	m := testModel()
	devices := []Device{headDevice(16 * gb), wired("peer", 0, 24*gb)}
	if _, ok := FitsOnOne(m, 4096, devices, Policy{}); ok {
		t.Fatal("a 34 GiB model must not report fitting on a 24 GiB device")
	}
}

// The headline case: a model too large for any node, held across three.
func TestBuildPoolsAcrossNodes(t *testing.T) {
	m := testModel()
	devices := []Device{
		headDevice(16 * gb),
		wired("peer-a", 0, 16*gb),
		wired("peer-b", 0, 16*gb),
	}
	out, err := Build(m, 4096, devices, head, Policy{})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if !out.Fits() {
		t.Fatalf("plan did not form; %d layers unplaced, %.1f GiB short",
			out.UnplacedLayers, gib(out.ShortfallBytes))
	}

	plan := out.Plan
	placed := 0
	for _, a := range plan.Assignments {
		placed += a.Layers
	}
	if placed != m.Layers {
		t.Fatalf("placed %d layers, want all %d", placed, m.Layers)
	}
	if !plan.Assignments[0].IsHead || plan.Assignments[0].Device.NodeUUID != head {
		t.Fatal("the first assignment must be the head node")
	}
	if len(plan.Nodes()) != 3 {
		t.Fatalf("used %d nodes, want 3", len(plan.Nodes()))
	}
	if plan.Boundaries != 2 {
		t.Fatalf("boundaries = %d, want 2 for three machines", plan.Boundaries)
	}
}

// Layer ranges are consecutive and contiguous, because layers are evaluated in
// order and a gap would cross the network twice for nothing.
func TestAssignmentsCoverLayersContiguously(t *testing.T) {
	m := testModel()
	devices := []Device{headDevice(16 * gb), wired("peer-a", 0, 16*gb), wired("peer-b", 0, 16*gb)}
	out, err := Build(m, 4096, devices, head, Policy{})
	if err != nil || !out.Fits() {
		t.Fatalf("Build: %v, fits=%v", err, out.Fits())
	}

	next := 0
	for _, a := range out.Plan.Assignments {
		if a.FirstLayer != next {
			t.Fatalf("assignment starts at %d, want %d", a.FirstLayer, next)
		}
		next += a.Layers
	}
	if next != m.Layers {
		t.Fatalf("layers end at %d, want %d", next, m.Layers)
	}
}

// Spreading layers evenly maximizes machine boundaries, and each boundary is a
// serialized round trip per token. The planner uses the fewest machines that fit.
func TestBuildUsesTheFewestMachinesThatFit(t *testing.T) {
	m := testModel()
	devices := []Device{
		headDevice(64 * gb), // can hold everything on its own
		wired("peer-a", 0, 64*gb),
		wired("peer-b", 0, 64*gb),
	}
	out, err := Build(m, 4096, devices, head, Policy{})
	if err != nil || !out.Fits() {
		t.Fatalf("Build: %v, fits=%v", err, out.Fits())
	}
	if got := len(out.Plan.Nodes()); got != 1 {
		t.Fatalf("used %d nodes for a model that fits on one, want 1", got)
	}
	if out.Plan.Boundaries != 0 {
		t.Fatalf("boundaries = %d, want none", out.Plan.Boundaries)
	}
}

// Filling a big wireless GPU before a smaller wired one produces a pool that
// holds the model and runs at the speed of the radio.
func TestBetterLinkIsFilledBeforeMoreMemory(t *testing.T) {
	m := testModel()
	devices := []Device{
		headDevice(4 * gb),
		wired("peer-lan", 0, 24*gb),
		wired("peer-wifi", 0, 80*gb),
	}
	devices[2].Link = linkq.ClassWiFi

	out, err := Build(m, 4096, devices, head, Policy{AllowWiFiDonors: true})
	if err != nil || !out.Fits() {
		t.Fatalf("Build: %v, fits=%v", err, out.Fits())
	}
	var lan, wifi int
	for _, a := range out.Plan.Assignments {
		switch a.Device.NodeUUID {
		case "peer-lan":
			lan = a.Layers
		case "peer-wifi":
			wifi = a.Layers
		}
	}
	if lan <= wifi {
		t.Fatalf("wired donor took %d layers and wireless took %d; the wired one should be filled first", lan, wifi)
	}
}

func TestSlowestLinkIgnoresTheHead(t *testing.T) {
	m := testModel()
	h := headDevice(64 * gb)
	h.Link = linkq.ClassUnknown // nothing measures a path to itself
	out, err := Build(m, 4096, []Device{h}, head, Policy{})
	if err != nil || !out.Fits() {
		t.Fatalf("Build: %v, fits=%v", err, out.Fits())
	}
	if out.Plan.SlowestLink != linkq.ClassLocal {
		t.Fatalf("SlowestLink = %v for a head-only plan, want local", out.Plan.SlowestLink)
	}
}

func TestDonorGates(t *testing.T) {
	m := testModel()
	cases := []struct {
		name    string
		mutate  func(*Device)
		policy  Policy
		wantWhy string
	}{
		{
			name:    "wide-area donors are refused outright",
			mutate:  func(d *Device) { d.Link = linkq.ClassWAN },
			wantWhy: "wide-area",
		},
		{
			name:    "unmeasured links are refused by default",
			mutate:  func(d *Device) { d.Link = linkq.ClassUnknown },
			wantWhy: "has not been measured",
		},
		{
			name:    "wireless donors are refused by default",
			mutate:  func(d *Device) { d.Link = linkq.ClassWiFi },
			wantWhy: "wireless",
		},
		{
			name:    "a flapping donor is refused however much memory it has",
			mutate:  func(d *Device) { d.Stability = 0.6 },
			wantWhy: "stability",
		},
		{
			name:    "unidentified media is refused under a strict policy",
			mutate:  func(d *Device) { d.MediaKnown = false },
			policy:  Policy{RequireKnownMedia: true},
			wantWhy: "media could not be positively identified",
		},
		{
			name:    "a device with no room after overhead is refused",
			mutate:  func(d *Device) { d.FreeBytes = 128 << 20 },
			wantWhy: "no usable VRAM",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			donor := wired("peer", 0, 64*gb)
			tc.mutate(&donor)
			out, err := Build(m, 4096, []Device{headDevice(4 * gb), donor}, head, tc.policy)
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			if out.Fits() {
				t.Fatal("the pool formed with a donor that should have been refused")
			}
			if len(out.Rejected) != 1 {
				t.Fatalf("rejections = %v, want exactly one", out.Rejected)
			}
			if !strings.Contains(out.Rejected[0].Reason, tc.wantWhy) {
				t.Fatalf("reason %q does not mention %q", out.Rejected[0].Reason, tc.wantWhy)
			}
		})
	}
}

// The head node's own devices are never gated on link class: they are on the
// machine running the server, and a measurement about a path that does not exist
// must not refuse the pool its own memory.
func TestHeadDevicesAreNotLinkGated(t *testing.T) {
	m := testModel()
	h := headDevice(64 * gb)
	h.Link = linkq.ClassWAN
	h.Stability = 0
	h.MediaKnown = false

	out, err := Build(m, 4096, []Device{h}, head, Policy{RequireKnownMedia: true})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if !out.Fits() {
		t.Fatalf("the head's own GPU was gated: %v", out.Rejected)
	}
}

// A rejection is returned even when the plan succeeds, because "the cluster has
// plenty free, why is that machine not being used" is the first question a user
// asks.
func TestRejectionsAreReportedOnSuccessToo(t *testing.T) {
	m := testModel()
	bad := wired("peer-wifi", 0, 64*gb)
	bad.Link = linkq.ClassWiFi

	out, err := Build(m, 4096, []Device{headDevice(64 * gb), bad}, head, Policy{})
	if err != nil || !out.Fits() {
		t.Fatalf("Build: %v, fits=%v", err, out.Fits())
	}
	if len(out.Rejected) != 1 {
		t.Fatalf("rejections = %v, want the wireless donor explained", out.Rejected)
	}
}

// Every donor is another serialized hop and another way for the pool to die, so
// the cap counts machines rather than GPUs: a second card on a machine already
// in the pool is free.
func TestMaxDonorsCountsMachinesNotDevices(t *testing.T) {
	m := testModel()
	devices := []Device{
		headDevice(8 * gb),
		wired("peer-a", 0, 16*gb),
		wired("peer-a", 1, 16*gb),
		wired("peer-b", 0, 16*gb),
		wired("peer-c", 0, 16*gb),
		wired("peer-d", 0, 16*gb),
	}
	out, err := Build(m, 4096, devices, head, Policy{MaxDonors: 2})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if !out.Fits() {
		t.Fatalf("plan did not form; %d layers unplaced", out.UnplacedLayers)
	}
	nodes := out.Plan.Nodes()
	if len(nodes) != 3 {
		t.Fatalf("used nodes %v, want the head plus two donors", nodes)
	}
	devicesUsed := len(out.Plan.Assignments)
	if devicesUsed < 4 {
		t.Fatalf("used %d devices; both of peer-a's GPUs should be available once it is in the pool", devicesUsed)
	}
}

func TestDoesNotFitReportsTheGap(t *testing.T) {
	m := testModel()
	out, err := Build(m, 4096, []Device{headDevice(8 * gb)}, head, Policy{})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if out.Fits() {
		t.Fatal("34 GiB should not fit in 8 GiB")
	}
	if out.UnplacedLayers <= 0 {
		t.Fatal("a non-fit should say how many layers had nowhere to go")
	}
	if out.ShortfallBytes == 0 {
		t.Fatal("a non-fit should estimate how much more memory was needed")
	}
}

// A head too small even for the embedding and output tensors cannot be helped by
// any donor, and the shortfall must include those tensors rather than only the
// layers.
func TestHeadTooSmallForNonRepeatingTensors(t *testing.T) {
	m := testModel()
	out, err := Build(m, 4096, []Device{headDevice(2 * gb), wired("peer", 0, 200*gb)}, head, Policy{})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if out.Fits() {
		t.Fatal("a head that cannot hold the output tensors cannot head a pool")
	}
	if out.ShortfallBytes <= uint64(out.UnplacedLayers)*m.PerLayerBytes(4096) {
		t.Fatal("shortfall should include the non-repeating tensors that were never placed")
	}
}

// Doubling the context does not touch the weights and doubles the cache, and it
// is the cache that decides whether the last device has room.
func TestContextLengthChangesWhetherItFits(t *testing.T) {
	m := testModel()
	m.KVBytesPerLayerPerToken = KVBytesPerLayerPerToken(1024, 2) // 4 KiB per layer per token
	devices := []Device{headDevice(24 * gb), wired("peer", 0, 24*gb)}

	short, err := Build(m, 4096, devices, head, Policy{})
	if err != nil {
		t.Fatalf("Build short: %v", err)
	}
	if !short.Fits() {
		t.Fatalf("should fit at a 4k context; %d layers unplaced", short.UnplacedLayers)
	}

	long, err := Build(m, 131072, devices, head, Policy{})
	if err != nil {
		t.Fatalf("Build long: %v", err)
	}
	if long.Fits() {
		t.Fatal("a 128k context should not fit in the same memory")
	}
}

func TestTensorSplitSumsToOne(t *testing.T) {
	m := testModel()
	devices := []Device{headDevice(16 * gb), wired("peer-a", 0, 16*gb), wired("peer-b", 0, 16*gb)}
	out, err := Build(m, 4096, devices, head, Policy{})
	if err != nil || !out.Fits() {
		t.Fatalf("Build: %v, fits=%v", err, out.Fits())
	}
	sum := 0.0
	for _, f := range out.Plan.TensorSplit {
		sum += f
	}
	if sum < 0.999 || sum > 1.001 {
		t.Fatalf("tensor split sums to %v, want 1", sum)
	}
	arg := out.Plan.TensorSplitArg()
	if strings.Count(arg, ",") != len(out.Plan.Assignments)-1 {
		t.Fatalf("TensorSplitArg %q does not have one entry per assignment", arg)
	}
}

// Planning is a pure function of its inputs, so the same cluster produces the
// same plan — which is what makes a pool reproducible in a bug report.
func TestPlanningIsDeterministic(t *testing.T) {
	m := testModel()
	devices := []Device{
		headDevice(12 * gb),
		wired("peer-b", 0, 16*gb),
		wired("peer-a", 0, 16*gb),
		wired("peer-c", 0, 16*gb),
	}
	first, err := Build(m, 4096, devices, head, Policy{})
	if err != nil || !first.Fits() {
		t.Fatalf("Build: %v", err)
	}
	for range 20 {
		again, err := Build(m, 4096, devices, head, Policy{})
		if err != nil || !again.Fits() {
			t.Fatalf("rebuild: %v", err)
		}
		if again.Plan.TensorSplitArg() != first.Plan.TensorSplitArg() {
			t.Fatalf("split changed between runs: %q then %q",
				first.Plan.TensorSplitArg(), again.Plan.TensorSplitArg())
		}
	}
}

// No assignment may exceed what the policy said the device could give.
func TestNoAssignmentExceedsItsDeviceBudget(t *testing.T) {
	m := testModel()
	m.KVBytesPerLayerPerToken = KVBytesPerLayerPerToken(512, 2)
	policy := Policy{}.withDefaults()
	devices := []Device{headDevice(20 * gb), wired("peer-a", 0, 12*gb), wired("peer-b", 0, 40*gb)}

	out, err := Build(m, 8192, devices, head, Policy{})
	if err != nil || !out.Fits() {
		t.Fatalf("Build: %v, fits=%v", err, out.Fits())
	}
	for _, a := range out.Plan.Assignments {
		budget := policy.usable(a.Device) + policy.PerDeviceOverheadBytes
		if a.TotalBytes > budget {
			t.Fatalf("%s was assigned %.2f GiB against a %.2f GiB budget",
				a.Device, gib(a.TotalBytes), gib(budget))
		}
	}
}

func TestHeadroomIsRespected(t *testing.T) {
	m := testModel()
	// Just enough at full capacity, and not enough once a tenth is held back
	// plus the compute-buffer overhead.
	devices := []Device{headDevice(m.TotalBytes(4096) + 1)}

	out, err := Build(m, 4096, devices, head, Policy{})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if out.Fits() {
		t.Fatal("a plan that uses every free byte should not form under default headroom")
	}

	relaxed, err := Build(m, 4096, devices, head, Policy{Headroom: 1, PerDeviceOverheadBytes: 1})
	if err != nil {
		t.Fatalf("Build relaxed: %v", err)
	}
	if !relaxed.Fits() {
		t.Fatal("the same model should fit once headroom and overhead are removed")
	}
}
