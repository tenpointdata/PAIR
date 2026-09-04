// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package linkq

import (
	"sync"
	"testing"
	"time"
)

// fixedClock returns a tracker whose clock is under the test's control, so
// freshness and ordering are assertions rather than races.
func fixedClock(t *testing.T) (*Tracker, func(time.Duration)) {
	t.Helper()
	now := time.Date(2026, 3, 1, 12, 0, 0, 0, time.UTC)
	tr := NewTracker()
	tr.now = func() time.Time { return now }
	return tr, func(d time.Duration) { now = now.Add(d) }
}

func TestTrackerReportsNothingForAnUnseenPeer(t *testing.T) {
	tr := NewTracker()
	if _, ok := tr.Path("never-seen"); ok {
		t.Fatal("a peer nothing was observed about must not report a path")
	}
}

// A blank node id cannot be attributed. Accepting one would pool every
// unattributable sample in the cluster into a single meaningless average.
func TestTrackerRefusesBlankNodeID(t *testing.T) {
	tr := NewTracker()
	tr.ObservePath("", TopologySameSubnet, MediaWired)
	tr.ObserveRTT("", 5*time.Millisecond)
	tr.ObserveTransfer("", 10<<20, time.Second)
	tr.ObserveReachable("", true)
	if got := len(tr.Snapshot()); got != 0 {
		t.Fatalf("snapshot has %d entries, want 0", got)
	}
}

func TestTrackerClassifiesFromPathAndPublishedMedia(t *testing.T) {
	tr, _ := fixedClock(t)
	tr.ObservePath("a", TopologySameSubnet, MediaWired)

	p, ok := tr.Path("a")
	if !ok {
		t.Fatal("path missing after observation")
	}
	if p.Class != ClassLAN {
		t.Fatalf("class = %v, want %v", p.Class, ClassLAN)
	}
	if p.MediaKnown {
		t.Fatal("media cannot be known while the remote end has published nothing")
	}

	tr.ObserveRemoteMedia("a", MediaWireless)
	p, _ = tr.Path("a")
	if p.Class != ClassWiFi {
		t.Fatalf("class after remote media = %v, want %v", p.Class, ClassWiFi)
	}
	if !p.MediaKnown {
		t.Fatal("media should be known once both ends are identified")
	}
}

// Learning a peer's class for the first time is this package catching up, not
// the network moving. Counting it would give every peer a phantom roam at
// startup — which is exactly the signal a pool donor gate reads.
func TestFirstClassificationIsNotAChange(t *testing.T) {
	tr, _ := fixedClock(t)
	tr.ObservePath("a", TopologySameSubnet, MediaWired)
	if p, _ := tr.Path("a"); p.ClassChanges != 0 {
		t.Fatalf("ClassChanges = %d after first observation, want 0", p.ClassChanges)
	}
}

func TestClassChangesCountRoaming(t *testing.T) {
	tr, _ := fixedClock(t)
	tr.ObservePath("a", TopologySameSubnet, MediaWired)    // lan
	tr.ObservePath("a", TopologySameSubnet, MediaWireless) // wifi
	tr.ObservePath("a", TopologyRelayed, MediaWireless)    // wan
	tr.ObservePath("a", TopologyRelayed, MediaWireless)    // wan again, no change

	p, _ := tr.Path("a")
	if p.ClassChanges != 2 {
		t.Fatalf("ClassChanges = %d, want 2", p.ClassChanges)
	}
	if p.Class != ClassWAN {
		t.Fatalf("class = %v, want %v", p.Class, ClassWAN)
	}
}

func TestRTTSeedsThenSmooths(t *testing.T) {
	tr, _ := fixedClock(t)
	tr.ObserveRTT("a", 100*time.Millisecond)
	if p, _ := tr.Path("a"); p.RTT != 100*time.Millisecond {
		t.Fatalf("first sample should seed exactly, got %v", p.RTT)
	}
	// One 200ms sample at alpha 0.25 moves the average a quarter of the way.
	tr.ObserveRTT("a", 200*time.Millisecond)
	p, _ := tr.Path("a")
	if p.RTT <= 100*time.Millisecond || p.RTT >= 200*time.Millisecond {
		t.Fatalf("RTT = %v, want it smoothed between the two samples", p.RTT)
	}
	if got, want := p.RTT, 125*time.Millisecond; got != want {
		t.Fatalf("RTT = %v, want %v", got, want)
	}
}

// A clamped outlier is still an invented measurement, so implausible samples are
// dropped rather than bounded.
func TestRTTRejectsImplausibleSamples(t *testing.T) {
	tr, _ := fixedClock(t)
	tr.ObserveRTT("a", 40*time.Millisecond)
	before, _ := tr.Path("a")

	tr.ObserveRTT("a", -time.Second)
	tr.ObserveRTT("a", 0)
	tr.ObserveRTT("a", maxPlausibleRTT+time.Second)

	after, _ := tr.Path("a")
	if after.RTT != before.RTT {
		t.Fatalf("RTT moved from %v to %v on rejected samples", before.RTT, after.RTT)
	}
	if after.Samples != before.Samples {
		t.Fatal("a rejected sample must not count as evidence")
	}
}

// A 200-byte reply that completes in a millisecond implies 200 kB/s, which
// describes the engine's think time, not the link.
func TestThroughputIgnoresSamplesTooSmallToMeanAnything(t *testing.T) {
	tr, _ := fixedClock(t)
	tr.ObserveTransfer("a", 200, time.Millisecond)
	tr.ObserveTransfer("a", minSampleBytes, minSampleDuration/2)
	if p, ok := tr.Path("a"); ok && p.Throughput != 0 {
		t.Fatalf("throughput = %d, want 0 from unusable samples", p.Throughput)
	}

	tr.ObserveTransfer("a", 10<<20, time.Second) // 10 MiB in 1s
	p, _ := tr.Path("a")
	if p.Throughput == 0 {
		t.Fatal("a real transfer should produce a rate")
	}
	if p.Throughput < 9<<20 || p.Throughput > 11<<20 {
		t.Fatalf("throughput = %d, want roughly 10 MiB/s", p.Throughput)
	}
}

// Stability forgives slowly on purpose: one failure costs more than one success
// recovers, because the cost of over-trusting a flapping donor is a dead pool.
func TestStabilityFallsFastAndRecoversSlowly(t *testing.T) {
	tr, _ := fixedClock(t)
	for range 20 {
		tr.ObserveReachable("a", true)
	}
	healthy, _ := tr.Path("a")
	if healthy.Stability < 0.99 {
		t.Fatalf("stability = %v after a clean run, want ~1", healthy.Stability)
	}

	tr.ObserveReachable("a", false)
	dropped, _ := tr.Path("a")
	fall := healthy.Stability - dropped.Stability

	tr.ObserveReachable("a", true)
	recovered, _ := tr.Path("a")
	rise := recovered.Stability - dropped.Stability

	if rise >= fall {
		t.Fatalf("recovery %v should be slower than the fall %v", rise, fall)
	}
}

func TestForgetDropsOnlyTheNamedPeer(t *testing.T) {
	tr, _ := fixedClock(t)
	tr.ObservePath("a", TopologySameSubnet, MediaWired)
	tr.ObservePath("b", TopologyRouted, MediaWired)

	tr.Forget("a")
	if _, ok := tr.Path("a"); ok {
		t.Fatal("forgotten peer still tracked")
	}
	if _, ok := tr.Path("b"); !ok {
		t.Fatal("Forget dropped an unrelated peer")
	}
}

func TestSnapshotIsACopy(t *testing.T) {
	tr, _ := fixedClock(t)
	tr.ObservePath("a", TopologySameSubnet, MediaWired)

	snap := tr.Snapshot()
	snap["a"] = Path{Class: ClassWAN}
	snap["injected"] = Path{}

	if p, _ := tr.Path("a"); p.Class != ClassLAN {
		t.Fatalf("mutating a snapshot changed the tracker: %v", p.Class)
	}
	if _, ok := tr.Path("injected"); ok {
		t.Fatal("mutating a snapshot added a peer to the tracker")
	}
}

func TestUpdatedAtAdvancesWithTheClock(t *testing.T) {
	tr, advance := fixedClock(t)
	tr.ObservePath("a", TopologySameSubnet, MediaWired)
	first, _ := tr.Path("a")

	advance(time.Minute)
	tr.ObserveRTT("a", 5*time.Millisecond)
	second, _ := tr.Path("a")

	if !second.UpdatedAt.After(first.UpdatedAt) {
		t.Fatalf("UpdatedAt did not advance: %v then %v", first.UpdatedAt, second.UpdatedAt)
	}
	if !second.Fresh(second.UpdatedAt, time.Second) {
		t.Fatal("a just-taken reading should be fresh")
	}
	if first.Fresh(second.UpdatedAt, 30*time.Second) {
		t.Fatal("a minute-old reading should be stale within thirty seconds")
	}
}

// Every consumer of this tracker is concurrent — proxies stream, the scanner
// sweeps, the cluster manager reconciles — so the race detector is the point of
// this test rather than the assertions.
func TestConcurrentObservationsAreSafe(t *testing.T) {
	tr := NewTracker()
	var wg sync.WaitGroup
	for i := range 8 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			id := string(rune('a' + i%3))
			for range 200 {
				tr.ObserveRTT(id, 10*time.Millisecond)
				tr.ObserveTransfer(id, 1<<20, 100*time.Millisecond)
				tr.ObserveReachable(id, i%2 == 0)
				tr.ObservePath(id, TopologySameSubnet, MediaWired)
				_, _ = tr.Path(id)
				_ = tr.Snapshot()
			}
		}(i)
	}
	wg.Wait()
	if got := len(tr.Snapshot()); got != 3 {
		t.Fatalf("tracked %d peers, want 3", got)
	}
}
