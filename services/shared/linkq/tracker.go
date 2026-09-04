// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package linkq

import (
	"sync"
	"time"
)

// Smoothing factors for the three tracked quantities. Each is chosen for how
// quickly acting on a wrong answer hurts, which is not the same for all three.
const (
	// rttAlpha weights each new round-trip sample. Responsive, because RTT is
	// the signal that changes when a peer roams onto a worse link, and the
	// sooner routing sees it the fewer requests are steered into it.
	rttAlpha = 0.25
	// throughputAlpha is slower. A single response can be small, chunked, or
	// throttled by the engine rather than the link, so one sample is weak
	// evidence about a link's capacity even when it passes the size floor below.
	throughputAlpha = 0.20
	// stabilityAlpha is slowest, and deliberately asymmetric in effect: a run of
	// successes climbs back to 1.0 far more slowly than a single failure falls
	// away from it. A donor that dropped a pool is not vindicated by answering
	// the next probe thirty seconds later, and the cost of over-trusting it is
	// another failed pool rather than another slow request.
	stabilityAlpha = 0.10
)

// Sample admission thresholds. A measurement that cannot mean anything is worse
// than no measurement, because it looks like evidence.
const (
	// minSampleBytes is the smallest transfer worth deriving a rate from. A
	// 200-byte response that completes in a millisecond implies 200 kB/s, which
	// describes the engine's think time, not the link.
	minSampleBytes = 64 * 1024
	// minSampleDuration is the shortest transfer worth deriving a rate from, for
	// the same reason from the other direction: below it, clock granularity is a
	// larger term than the network.
	minSampleDuration = 20 * time.Millisecond
	// maxPlausibleRTT rejects a sample that measured something other than the
	// network — a peer paused by its own inference load, a laptop resuming from
	// sleep mid-request, a debugger. Beyond this the number says nothing about
	// the link, and folding it into the average would poison a peer's reading
	// for minutes.
	maxPlausibleRTT = 30 * time.Second
)

// Tracker accumulates what has been observed about the link to each peer and
// answers with a smoothed Path.
//
// It is a sink, not a prober. Every Observe method is called by code that was
// already sending or receiving something for its own reasons: the cluster
// manager's roster reconcile, the scanner's telemetry sweep, a proxy streaming a
// generation back. Nothing here originates traffic, which is what makes the
// measurement free and what keeps it honest on a metered link.
//
// The zero value is not usable; call NewTracker.
type Tracker struct {
	mu    sync.RWMutex
	peers map[string]*peerState

	// now reads the clock. Overridden in tests.
	now func() time.Time
}

// peerState is the accumulated evidence for one peer. Class is stored rather
// than derived on read so a change can be counted as it happens; deriving it on
// every read would make ClassChanges depend on how often somebody looked.
type peerState struct {
	topology     Topology
	localMedia   Media
	remoteMedia  Media
	class        Class
	rtt          time.Duration
	throughput   uint64
	stability    float64
	haveStab     bool
	classChanges int
	samples      int
	updatedAt    time.Time
}

// NewTracker returns an empty Tracker.
func NewTracker() *Tracker {
	return &Tracker{peers: make(map[string]*peerState), now: time.Now}
}

// state returns nodeID's accumulator, creating it on first observation. Callers
// hold the write lock. A blank id is refused: an observation that cannot be
// attributed would credit or blame an arbitrary peer, and a map entry under ""
// accumulates every unattributable sample in the cluster into one meaningless
// average.
func (t *Tracker) state(nodeID string) *peerState {
	if nodeID == "" {
		return nil
	}
	st, ok := t.peers[nodeID]
	if !ok {
		st = &peerState{}
		t.peers[nodeID] = st
	}
	return st
}

// reclassify recomputes a peer's class and counts a transition. Callers hold the
// write lock.
//
// The first classification is not a change. A peer moving from ClassUnknown to
// its real class is this package learning, not the network moving, and counting
// it would give every peer in the cluster one phantom roam at startup — exactly
// the signal a pool donor gate is reading.
func (st *peerState) reclassify() {
	next := Classify(st.topology, st.localMedia, st.remoteMedia)
	if st.class != ClassUnknown && next != st.class {
		st.classChanges++
	}
	st.class = next
}

// ObservePath records how a peer was reached and what local interface carried
// it. Call it when a connection is established or when the chosen endpoint
// changes — a roam, a failover to a relay, a switch from wireless to a docking
// station.
func (t *Tracker) ObservePath(nodeID string, topology Topology, localMedia Media) {
	t.mu.Lock()
	defer t.mu.Unlock()
	st := t.state(nodeID)
	if st == nil {
		return
	}
	st.topology = topology
	st.localMedia = localMedia
	st.reclassify()
	st.samples++
	st.updatedAt = t.now()
}

// ObserveRemoteMedia records what a peer published about its own interface. It
// is a claim rather than a measurement — only the peer can see its own hardware
// — but it is a claim made over an authenticated channel by a node with no
// incentive to lie, and the alternative is having no idea whether the far end is
// on a radio.
func (t *Tracker) ObserveRemoteMedia(nodeID string, m Media) {
	t.mu.Lock()
	defer t.mu.Unlock()
	st := t.state(nodeID)
	if st == nil {
		return
	}
	st.remoteMedia = m
	st.reclassify()
	st.updatedAt = t.now()
}

// ObserveRTT folds one round-trip measurement into a peer's smoothed RTT.
// Non-positive and implausibly long samples are discarded rather than clamped:
// a clamped outlier is still an invented measurement.
func (t *Tracker) ObserveRTT(nodeID string, d time.Duration) {
	if d <= 0 || d > maxPlausibleRTT {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	st := t.state(nodeID)
	if st == nil {
		return
	}
	if st.rtt == 0 {
		st.rtt = d
	} else {
		st.rtt = time.Duration(rttAlpha*float64(d) + (1-rttAlpha)*float64(st.rtt))
	}
	st.samples++
	st.updatedAt = t.now()
}

// ObserveTransfer folds one completed transfer into a peer's smoothed
// throughput. Samples too small or too brief to describe the link are ignored;
// see minSampleBytes.
func (t *Tracker) ObserveTransfer(nodeID string, bytes uint64, d time.Duration) {
	if bytes < minSampleBytes || d < minSampleDuration {
		return
	}
	rate := uint64(float64(bytes) / d.Seconds())
	if rate == 0 {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	st := t.state(nodeID)
	if st == nil {
		return
	}
	if st.throughput == 0 {
		st.throughput = rate
	} else {
		st.throughput = uint64(throughputAlpha*float64(rate) + (1-throughputAlpha)*float64(st.throughput))
	}
	st.samples++
	st.updatedAt = t.now()
}

// ObserveReachable folds one reachability outcome into a peer's stability. Pass
// the result of something that actually tried to reach the peer — a completed
// request, a refused dial, a handshake that timed out — not the result of asking
// whether it is currently listed as present.
func (t *Tracker) ObserveReachable(nodeID string, ok bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	st := t.state(nodeID)
	if st == nil {
		return
	}
	v := 0.0
	if ok {
		v = 1.0
	}
	if !st.haveStab {
		st.stability = v
		st.haveStab = true
	} else {
		st.stability = stabilityAlpha*v + (1-stabilityAlpha)*st.stability
	}
	st.samples++
	st.updatedAt = t.now()
}

// snapshot renders a peer's accumulator. Callers hold at least the read lock.
func (st *peerState) snapshot() Path {
	return Path{
		Class:        st.class,
		LocalMedia:   st.localMedia,
		RemoteMedia:  st.remoteMedia,
		MediaKnown:   st.localMedia != MediaUnknown && st.remoteMedia != MediaUnknown,
		Topology:     st.topology,
		RTT:          st.rtt,
		Throughput:   st.throughput,
		Stability:    st.stability,
		ClassChanges: st.classChanges,
		Samples:      st.samples,
		UpdatedAt:    st.updatedAt,
	}
}

// Path returns what is known about the link to nodeID. ok is false for a peer
// nothing has been observed about, which is a different answer from a peer known
// to be unreachable and must not be collapsed into one.
func (t *Tracker) Path(nodeID string) (Path, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	st, ok := t.peers[nodeID]
	if !ok {
		return Path{}, false
	}
	return st.snapshot(), true
}

// Snapshot returns every tracked peer's Path, for relaying to the scheduler or
// rendering in the interface.
func (t *Tracker) Snapshot() map[string]Path {
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make(map[string]Path, len(t.peers))
	for id, st := range t.peers {
		out[id] = st.snapshot()
	}
	return out
}

// Forget drops a peer's accumulated evidence. Call it when a node leaves the
// cluster or is removed, not when it merely goes quiet: a peer that comes back
// after a reboot is the same machine on the same link, and making it re-earn a
// reading it already has costs a warm-up every restart.
func (t *Tracker) Forget(nodeID string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.peers, nodeID)
}

// Penalty is the rank cost of a class, in the same units nvpair-job-scheduler
// already ranks in — one unit is one pending job, or one step of GPU pressure.
//
// Sharing the unit is the point: the scheduler sorts by pending plus pressure,
// so a link term expressed in anything else would need a weight, and a weight
// would need tuning against hardware nobody has yet. Expressed in jobs it reads
// directly: a Wi-Fi peer must look one job emptier than a wired one to win, and
// a WAN peer three.
//
// ClassUnknown costs one rather than nothing. A peer nothing has been measured
// about should not beat a peer measured to be on the same switch, and it should
// not be excluded either — it has done nothing wrong except be quiet.
func (c Class) Penalty() int {
	switch c {
	case ClassLocal, ClassLAN:
		return 0
	case ClassWiFi:
		return 1
	case ClassWAN:
		return 3
	default:
		return 1
	}
}
