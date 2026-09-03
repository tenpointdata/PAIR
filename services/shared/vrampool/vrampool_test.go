// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package vrampool

import (
	"strings"
	"testing"

	"nvpair-shared/linkq"
)

func TestKVBytesPerLayerPerTokenChargesKeysAndValues(t *testing.T) {
	// A grouped-query model with a 1024-wide KV projection at two bytes an
	// element: 2 x 1024 x 2 = 4 KiB per layer per token.
	if got, want := KVBytesPerLayerPerToken(1024, 2), uint64(4096); got != want {
		t.Fatalf("KVBytesPerLayerPerToken = %d, want %d", got, want)
	}
}

func TestPerLayerWeightBytesExcludesNonRepeatingTensors(t *testing.T) {
	m := Model{Layers: 32, WeightBytes: 34 * gb, NonRepeatingBytes: 2 * gb}
	if got, want := m.PerLayerWeightBytes(), gb; got != want {
		t.Fatalf("PerLayerWeightBytes = %d, want %d", got, want)
	}
}

func TestPerLayerBytesAddsTheCacheForTheWholeContext(t *testing.T) {
	m := Model{
		Layers:                  32,
		WeightBytes:             34 * gb,
		NonRepeatingBytes:       2 * gb,
		KVBytesPerLayerPerToken: 4096,
	}
	if got, want := m.PerLayerBytes(0), gb; got != want {
		t.Fatalf("PerLayerBytes(0) = %d, want the weights alone (%d)", got, want)
	}
	if got, want := m.PerLayerBytes(4096), gb+4096*4096; got != want {
		t.Fatalf("PerLayerBytes(4096) = %d, want %d", got, want)
	}
	// Weights are fixed; the cache is what a longer context costs.
	if m.PerLayerBytes(8192)-m.PerLayerBytes(4096) != 4096*4096 {
		t.Fatal("doubling the context should add exactly one context of cache")
	}
}

func TestTotalBytesCoversEveryLayerAndTheHeadTensors(t *testing.T) {
	m := Model{Layers: 4, WeightBytes: 6 * gb, NonRepeatingBytes: 2 * gb, KVBytesPerLayerPerToken: 1024}
	// Four 1 GiB layers plus 2 GiB of non-repeating tensors, plus cache.
	want := 2*gb + 4*(gb+1024*2048)
	if got := m.TotalBytes(2048); got != want {
		t.Fatalf("TotalBytes = %d, want %d", got, want)
	}
}

// A malformed description must be rejected rather than silently producing a
// plan that overcommits every device in the cluster.
func TestModelValidation(t *testing.T) {
	cases := map[string]Model{
		"no layer count":        {Name: "x", WeightBytes: gb},
		"no weights":            {Name: "x", Layers: 8},
		"impossible head share": {Name: "x", Layers: 8, WeightBytes: gb, NonRepeatingBytes: 2 * gb},
	}
	for name, m := range cases {
		if err := m.valid(); err == nil {
			t.Errorf("%s should be invalid", name)
		}
	}
	good := Model{Name: "x", Layers: 8, WeightBytes: 9 * gb, NonRepeatingBytes: gb}
	if err := good.valid(); err != nil {
		t.Errorf("a well-formed model was rejected: %v", err)
	}
}

func TestPolicyDefaultsFillZeroFields(t *testing.T) {
	p := Policy{}.withDefaults()
	if p.Headroom != DefaultHeadroom {
		t.Errorf("Headroom = %v, want %v", p.Headroom, DefaultHeadroom)
	}
	if p.PerDeviceOverheadBytes != DefaultPerDeviceOverhead {
		t.Errorf("PerDeviceOverheadBytes = %v, want %v", p.PerDeviceOverheadBytes, DefaultPerDeviceOverhead)
	}
	if p.MaxDonors != DefaultMaxDonors {
		t.Errorf("MaxDonors = %v, want %v", p.MaxDonors, DefaultMaxDonors)
	}
	if p.MinStability != DefaultMinStability {
		t.Errorf("MinStability = %v, want %v", p.MinStability, DefaultMinStability)
	}
}

// A headroom outside (0,1] is a mistake, not an instruction: 1.5 would plan
// against memory the device does not have.
func TestPolicyRejectsNonsensicalHeadroom(t *testing.T) {
	for _, bad := range []float64{-1, 0, 1.5, 100} {
		if got := (Policy{Headroom: bad}).withDefaults().Headroom; got != DefaultHeadroom {
			t.Errorf("Headroom %v survived as %v, want the default", bad, got)
		}
	}
	if got := (Policy{Headroom: 0.5}).withDefaults().Headroom; got != 0.5 {
		t.Errorf("a valid headroom was overwritten: %v", got)
	}
}

// These are unsigned, and a small device can owe more overhead than it has
// memory, which must saturate at zero rather than wrap to sixteen exabytes.
func TestUsableSaturatesRatherThanUnderflowing(t *testing.T) {
	p := Policy{}.withDefaults()
	tiny := Device{FreeBytes: 64 << 20} // well under the compute-buffer overhead
	if got := p.usable(tiny); got != 0 {
		t.Fatalf("usable on a tiny device = %d, want 0", got)
	}
	if got := p.usable(Device{FreeBytes: 0}); got != 0 {
		t.Fatalf("usable on an empty device = %d, want 0", got)
	}
}

func TestUsableAppliesHeadroomThenOverhead(t *testing.T) {
	p := Policy{Headroom: 0.5, PerDeviceOverheadBytes: gb}.withDefaults()
	if got, want := p.usable(Device{FreeBytes: 10 * gb}), 4*gb; got != want {
		t.Fatalf("usable = %d, want %d (half of ten, less one)", got, want)
	}
}

func TestDeviceStringNamesWhatAUserWouldRecognize(t *testing.T) {
	d := Device{NodeName: "studio", Index: 1, Name: "NVIDIA GeForce RTX 5080", FreeBytes: 12 * gb, Link: linkq.ClassWiFi}
	s := d.String()
	for _, want := range []string{"studio", "gpu1", "RTX 5080", "12.0 GiB", "wifi"} {
		if !strings.Contains(s, want) {
			t.Errorf("Device.String() = %q, missing %q", s, want)
		}
	}
}
