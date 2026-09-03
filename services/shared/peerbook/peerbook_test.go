// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package peerbook

import (
	"encoding/json"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

const peerA = "1111aaaa-2222-bbbb-3333-cccc4444dddd"

// clocked returns a book whose clock the test drives, because every ranking
// decision in this package is a function of elapsed time.
func clocked(t *testing.T) (*Book, func(time.Duration)) {
	t.Helper()
	now := time.Date(2026, 3, 1, 9, 0, 0, 0, time.UTC)
	b := New()
	b.now = func() time.Time { return now }
	return b, func(d time.Duration) { now = now.Add(d) }
}

func TestKindRoundTripsThroughText(t *testing.T) {
	for _, k := range []Kind{KindUnknown, KindLocal, KindName, KindMapped, KindReflexive, KindRelay} {
		if got := ParseKind(k.String()); got != k {
			t.Fatalf("ParseKind(%q) = %v, want %v", k.String(), got, k)
		}
	}
	if got := ParseKind("teleport"); got != KindUnknown {
		t.Fatalf("an unrecognized kind should read as unknown, got %v", got)
	}
}

func TestBookRefusesBlankUUID(t *testing.T) {
	b, _ := clocked(t)
	b.SetIdentity("", "nameless", map[string]int{"ni": 14318})
	b.Learn("", []Endpoint{{Kind: KindLocal, Host: "10.0.0.9"}}, "mdns")
	if got := len(b.Peers()); got != 0 {
		t.Fatalf("book holds %d peers, want 0", got)
	}
}

func TestCandidatesRenderHostAndPort(t *testing.T) {
	b, _ := clocked(t)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318, "cl": 14321})
	b.Learn(peerA, []Endpoint{
		{Kind: KindLocal, Host: "10.0.0.9"},
		{Kind: KindName, Host: "studio.example.net"},
		{Kind: KindLocal, Host: "fd00::9"},
	}, "mdns")

	got := b.Candidates(peerA, "cl")
	if !slices.Contains(got, "10.0.0.9:14321") {
		t.Errorf("missing IPv4 candidate in %v", got)
	}
	if !slices.Contains(got, "studio.example.net:14321") {
		t.Errorf("missing name candidate in %v", got)
	}
	if !slices.Contains(got, "[fd00::9]:14321") {
		t.Errorf("IPv6 literal must be bracketed, got %v", got)
	}
}

// A service the peer never advertised has no port, and inventing a default would
// manufacture a candidate whose only possible outcome is a timeout.
func TestCandidatesSkipUnadvertisedServices(t *testing.T) {
	b, _ := clocked(t)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	b.Learn(peerA, []Endpoint{{Kind: KindLocal, Host: "10.0.0.9"}}, "mdns")

	if got := b.Candidates(peerA, "ec"); len(got) != 0 {
		t.Fatalf("Candidates for an unadvertised service = %v, want none", got)
	}
	if got := b.Candidates(peerA, "ni"); len(got) != 1 {
		t.Fatalf("Candidates for an advertised service = %v, want one", got)
	}
}

// A translated path does not listen on the port the peer thinks it listens on.
func TestEndpointPortOverrideWins(t *testing.T) {
	b, _ := clocked(t)
	b.SetIdentity(peerA, "studio", map[string]int{"cl": 14321})
	b.Learn(peerA, []Endpoint{
		{Kind: KindMapped, Host: "198.51.100.7", Ports: map[string]int{"cl": 42021}},
	}, "manual")

	got := b.Candidates(peerA, "cl")
	if len(got) != 1 || got[0] != "198.51.100.7:42021" {
		t.Fatalf("Candidates = %v, want the endpoint's own port", got)
	}
}

// The core ranking claim from the design: evidence outranks a claim, whatever
// subnet either address is on.
func TestSuccessOutranksAnUntriedEndpoint(t *testing.T) {
	b, advance := clocked(t)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	b.Learn(peerA, []Endpoint{
		{Kind: KindLocal, Host: "10.0.0.9"},
		{Kind: KindName, Host: "studio.example.net"},
	}, "mdns")

	// Untried, so the kind tiebreak decides and the interface address leads.
	if got := b.Candidates(peerA, "ni"); got[0] != "10.0.0.9:14318" {
		t.Fatalf("untried order = %v, want the local address first", got)
	}

	b.RecordSuccess(peerA, "studio.example.net:14318")
	advance(time.Minute)

	if got := b.Candidates(peerA, "ni"); got[0] != "studio.example.net:14318" {
		t.Fatalf("after a success, order = %v, want the proven address first", got)
	}
}

// Three failures put an address behind everything untried, so a peer that moved
// stops costing a dial timeout on every lookup — but it stays in the list,
// because it is where that machine lives when it is at home.
func TestRepeatedFailuresDemoteButDoNotRemove(t *testing.T) {
	b, _ := clocked(t)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	b.Learn(peerA, []Endpoint{
		{Kind: KindLocal, Host: "10.0.0.9"},
		{Kind: KindName, Host: "studio.example.net"},
	}, "mdns")

	for range failureDemotion {
		b.RecordFailure(peerA, "10.0.0.9:14318")
	}
	got := b.Candidates(peerA, "ni")
	if len(got) != 2 {
		t.Fatalf("Candidates = %v, want both endpoints retained", got)
	}
	if got[0] != "studio.example.net:14318" || got[1] != "10.0.0.9:14318" {
		t.Fatalf("order = %v, want the failing address last", got)
	}
}

// One answer proves the address works now; a penalty earned on a network the
// peer has since left must not outlive the evidence against it.
func TestSuccessClearsTheFailureCount(t *testing.T) {
	b, _ := clocked(t)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	b.Learn(peerA, []Endpoint{{Kind: KindLocal, Host: "10.0.0.9"}}, "mdns")

	for range 5 {
		b.RecordFailure(peerA, "10.0.0.9")
	}
	if p, _ := b.Peer(peerA); p.Endpoints[0].Failures != 5 {
		t.Fatalf("failures = %d, want 5", p.Endpoints[0].Failures)
	}

	b.RecordSuccess(peerA, "10.0.0.9")
	p, _ := b.Peer(peerA)
	if p.Endpoints[0].Failures != 0 {
		t.Fatalf("failures = %d after a success, want 0", p.Endpoints[0].Failures)
	}
	if p.Endpoints[0].Successes != 1 {
		t.Fatalf("successes = %d, want 1", p.Endpoints[0].Successes)
	}
}

// A stale success eventually stops leading, because the network it proved
// something about is one the peer may have left.
func TestProvenSuccessExpires(t *testing.T) {
	b, advance := clocked(t)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	b.Learn(peerA, []Endpoint{
		{Kind: KindName, Host: "studio.example.net"},
		{Kind: KindLocal, Host: "10.0.0.9"},
	}, "mdns")
	b.RecordSuccess(peerA, "studio.example.net")

	advance(provenSuccess + time.Hour)
	// Re-advertise the untried one so it is current, then confirm it now leads.
	b.Learn(peerA, []Endpoint{{Kind: KindLocal, Host: "10.0.0.9"}}, "mdns")

	if got := b.Candidates(peerA, "ni"); got[0] != "10.0.0.9:14318" {
		t.Fatalf("order = %v, want the day-old success to have stopped leading", got)
	}
}

// An advertisement says where the peer is now, not that everywhere it has been
// is gone.
func TestLearnMergesRatherThanReplaces(t *testing.T) {
	b, _ := clocked(t)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	b.Learn(peerA, []Endpoint{{Kind: KindLocal, Host: "10.0.0.9"}}, "mdns")
	b.Learn(peerA, []Endpoint{{Kind: KindLocal, Host: "192.168.5.20"}}, "gossip:other")

	got := b.Candidates(peerA, "ni")
	if len(got) != 2 {
		t.Fatalf("Candidates = %v, want both the old and new address", got)
	}
}

// Re-learning the same endpoint updates it in place. Anything else accumulates a
// row per advertisement until the cap evicts real history.
func TestLearnDoesNotDuplicate(t *testing.T) {
	b, advance := clocked(t)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	for range 10 {
		b.Learn(peerA, []Endpoint{{Kind: KindLocal, Host: "10.0.0.9"}}, "mdns")
		advance(time.Second)
	}
	p, _ := b.Peer(peerA)
	if len(p.Endpoints) != 1 {
		t.Fatalf("endpoints = %d, want 1", len(p.Endpoints))
	}
	if p.Endpoints[0].FirstSeen.Equal(p.Endpoints[0].LastAdvertised) {
		t.Fatal("re-advertising should move LastAdvertised without moving FirstSeen")
	}
}

// Two kinds naming the same host are two claims about different paths — a local
// interface address and a port-forward can share a literal — so they stay
// distinct.
func TestEndpointIdentityIncludesKindAndRelay(t *testing.T) {
	b, _ := clocked(t)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	b.Learn(peerA, []Endpoint{
		{Kind: KindLocal, Host: "198.51.100.7"},
		{Kind: KindMapped, Host: "198.51.100.7"},
		{Kind: KindRelay, Host: "10.0.0.2", Via: "peer-b"},
		{Kind: KindRelay, Host: "10.0.0.2", Via: "peer-c"},
	}, "gossip:x")

	p, _ := b.Peer(peerA)
	if len(p.Endpoints) != 4 {
		t.Fatalf("endpoints = %d, want 4 distinct entries", len(p.Endpoints))
	}
}

// A relay with no relaying member is unusable and unrepairable, so it is refused
// rather than stored.
func TestRelayWithoutViaIsRefused(t *testing.T) {
	b, _ := clocked(t)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	b.Learn(peerA, []Endpoint{{Kind: KindRelay, Host: "10.0.0.2"}}, "gossip:x")
	if p, ok := b.Peer(peerA); ok && len(p.Endpoints) != 0 {
		t.Fatalf("endpoints = %v, want the unusable relay refused", p.Endpoints)
	}
}

// Using a relay spends a third machine's bandwidth, so it is never mixed into a
// concurrent probe sweep of direct addresses.
func TestRelaysAreExcludedFromCandidates(t *testing.T) {
	b, _ := clocked(t)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	b.Learn(peerA, []Endpoint{
		{Kind: KindRelay, Host: "10.0.0.2", Via: "peer-b"},
		{Kind: KindLocal, Host: "10.0.0.9"},
	}, "gossip:x")

	for _, addr := range b.Candidates(peerA, "ni") {
		if addr == "10.0.0.2:14318" {
			t.Fatal("a relay endpoint leaked into the direct candidate list")
		}
	}
	relays := b.Relays(peerA)
	if len(relays) != 1 || relays[0].Via != "peer-b" {
		t.Fatalf("Relays = %v, want the one relay", relays)
	}
}

func TestCandidatesAreCapped(t *testing.T) {
	b, _ := clocked(t)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	var eps []Endpoint
	for i := range MaxEndpointsPerPeer {
		eps = append(eps, Endpoint{Kind: KindLocal, Host: "10.0.0." + string(rune('a'+i))})
	}
	b.Learn(peerA, eps, "mdns")
	if got := len(b.Candidates(peerA, "ni")); got != MaxCandidates {
		t.Fatalf("Candidates returned %d, want the cap of %d", got, MaxCandidates)
	}
}

// Eviction must keep the address that works. Dropping by age would discard a
// static home address in favour of whatever a hotel handed out this morning.
func TestEndpointCapEvictsTheWorstNotTheOldest(t *testing.T) {
	b, advance := clocked(t)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	b.Learn(peerA, []Endpoint{{Kind: KindName, Host: "home.example.net"}}, "manual")
	b.RecordSuccess(peerA, "home.example.net")
	advance(time.Minute)

	for i := range MaxEndpointsPerPeer + 8 {
		b.Learn(peerA, []Endpoint{{Kind: KindLocal, Host: "203.0.113." + strconv.Itoa(i)}}, "mdns")
	}

	p, _ := b.Peer(peerA)
	if len(p.Endpoints) != MaxEndpointsPerPeer {
		t.Fatalf("endpoints = %d, want the cap of %d", len(p.Endpoints), MaxEndpointsPerPeer)
	}
	found := false
	for _, e := range p.Endpoints {
		if e.Host == "home.example.net" {
			found = true
		}
	}
	if !found {
		t.Fatal("eviction dropped the only address known to work")
	}
}

func TestPruneDropsOnlyStaleEndpoints(t *testing.T) {
	b, advance := clocked(t)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	b.Learn(peerA, []Endpoint{{Kind: KindLocal, Host: "10.0.0.9"}}, "mdns")

	advance(staleEndpoint + time.Hour)
	b.Learn(peerA, []Endpoint{{Kind: KindName, Host: "studio.example.net"}}, "manual")
	b.Prune()

	p, ok := b.Peer(peerA)
	if !ok {
		t.Fatal("pruning removed the peer itself; it should only remove endpoints")
	}
	if len(p.Endpoints) != 1 || p.Endpoints[0].Host != "studio.example.net" {
		t.Fatalf("endpoints after prune = %v, want only the fresh one", p.Endpoints)
	}
}

func TestSnapshotsAreCopies(t *testing.T) {
	b, _ := clocked(t)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	b.Learn(peerA, []Endpoint{{Kind: KindLocal, Host: "10.0.0.9", Ports: map[string]int{"ni": 1}}}, "mdns")

	p, _ := b.Peer(peerA)
	p.Endpoints[0].Host = "mutated"
	p.Endpoints[0].Ports["ni"] = 999
	p.Ports["ni"] = 999

	again, _ := b.Peer(peerA)
	if again.Endpoints[0].Host != "10.0.0.9" || again.Endpoints[0].Ports["ni"] != 1 || again.Ports["ni"] != 14318 {
		t.Fatalf("mutating a snapshot changed the book: %+v", again)
	}
}

func TestHostOf(t *testing.T) {
	cases := map[string]string{
		"10.0.0.9:14318":   "10.0.0.9",
		"10.0.0.9":         "10.0.0.9",
		"[fd00::9]:14318":  "fd00::9",
		"fd00::9":          "fd00::9",
		"host.example:443": "host.example",
		"host.example":     "host.example",
		"  10.0.0.9  ":     "10.0.0.9",
		"":                 "",
	}
	for in, want := range cases {
		if got := hostOf(in); got != want {
			t.Errorf("hostOf(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestEndpointJSONUsesKindSpelling(t *testing.T) {
	b, _ := clocked(t)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	b.Learn(peerA, []Endpoint{{Kind: KindReflexive, Host: "198.51.100.7"}}, "observed")
	p, _ := b.Peer(peerA)

	data, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back Peer
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.Endpoints[0].Kind != KindReflexive {
		t.Fatalf("kind survived as %v, want reflexive", back.Endpoints[0].Kind)
	}
	if want := `"kind":"reflexive"`; !strings.Contains(string(data), want) {
		t.Fatalf("encoded form %s does not contain %s", data, want)
	}
}
