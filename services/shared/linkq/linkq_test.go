// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package linkq

import (
	"encoding/json"
	"testing"
	"time"
)

func TestClassifyTopologySetsTheBase(t *testing.T) {
	cases := []struct {
		name     string
		topology Topology
		local    Media
		remote   Media
		want     Class
	}{
		{"unknown topology stays unknown", TopologyUnknown, MediaWired, MediaWired, ClassUnknown},
		{"loopback topology", TopologyLoopback, MediaUnknown, MediaUnknown, ClassLocal},
		{"loopback media either side", TopologySameSubnet, MediaLoopback, MediaWired, ClassLocal},
		{"wired on one subnet", TopologySameSubnet, MediaWired, MediaWired, ClassLAN},
		{"routed is wan even when wired", TopologyRouted, MediaWired, MediaWired, ClassWAN},
		{"relayed is wan", TopologyRelayed, MediaWired, MediaWired, ClassWAN},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Classify(tc.topology, tc.local, tc.remote); got != tc.want {
				t.Fatalf("Classify(%v,%v,%v) = %v, want %v", tc.topology, tc.local, tc.remote, got, tc.want)
			}
		})
	}
}

// Media may only make a class worse. This is the rule that lets a platform with
// no detector still rank its peers instead of collapsing every one of them to
// ClassUnknown, so it is worth pinning in both directions.
func TestClassifyMediaOnlyDemotes(t *testing.T) {
	cases := []struct {
		name   string
		local  Media
		remote Media
		want   Class
	}{
		{"unknown media leaves the topology answer alone", MediaUnknown, MediaUnknown, ClassLAN},
		{"one unknown end still leaves it alone", MediaWired, MediaUnknown, ClassLAN},
		{"wireless locally", MediaWireless, MediaWired, ClassWiFi},
		{"wireless remotely", MediaWired, MediaWireless, ClassWiFi},
		{"wireless both ends", MediaWireless, MediaWireless, ClassWiFi},
		{"virtual is not demoted; it is not a quality claim", MediaVirtual, MediaWired, ClassLAN},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Classify(TopologySameSubnet, tc.local, tc.remote); got != tc.want {
				t.Fatalf("Classify(sameSubnet,%v,%v) = %v, want %v", tc.local, tc.remote, got, tc.want)
			}
		})
	}
}

// Cellular skips ClassWiFi entirely: what disqualifies a tether is that it is
// metered and carrier-NATted, which is true on the same subnet as anywhere else.
func TestClassifyCellularIsAlwaysWAN(t *testing.T) {
	for _, topo := range []Topology{TopologySameSubnet, TopologyRouted, TopologyRelayed} {
		if got := Classify(topo, MediaCellular, MediaWired); got != ClassWAN {
			t.Fatalf("local cellular over %v = %v, want %v", topo, got, ClassWAN)
		}
		if got := Classify(topo, MediaWired, MediaCellular); got != ClassWAN {
			t.Fatalf("remote cellular over %v = %v, want %v", topo, got, ClassWAN)
		}
	}
}

func TestClassAndMediaRoundTripThroughText(t *testing.T) {
	for _, c := range []Class{ClassUnknown, ClassLocal, ClassLAN, ClassWiFi, ClassWAN} {
		if got := ParseClass(c.String()); got != c {
			t.Fatalf("ParseClass(%q) = %v, want %v", c.String(), got, c)
		}
	}
	for _, m := range []Media{MediaUnknown, MediaLoopback, MediaWired, MediaWireless, MediaCellular, MediaVirtual} {
		if got := ParseMedia(m.String()); got != m {
			t.Fatalf("ParseMedia(%q) = %v, want %v", m.String(), got, m)
		}
	}
}

// A spelling this build has never seen must read as unknown rather than fail the
// record: the rest of a peer's state is still usable.
func TestParseToleratesUnrecognizedSpellings(t *testing.T) {
	if got := ParseClass("satellite"); got != ClassUnknown {
		t.Fatalf("ParseClass(satellite) = %v, want %v", got, ClassUnknown)
	}
	if got := ParseMedia("li-fi"); got != MediaUnknown {
		t.Fatalf("ParseMedia(li-fi) = %v, want %v", got, MediaUnknown)
	}
	if got := ParseClass("  WiFi  "); got != ClassWiFi {
		t.Fatalf("ParseClass with padding and case = %v, want %v", got, ClassWiFi)
	}
}

// The wire carries spellings, not ordinals, so inserting a constant later is not
// a breaking change.
func TestJSONCarriesSpellingsNotOrdinals(t *testing.T) {
	b, err := json.Marshal(struct {
		Class Class `json:"class"`
		Media Media `json:"media"`
	}{ClassWiFi, MediaWireless})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if want := `{"class":"wifi","media":"wireless"}`; string(b) != want {
		t.Fatalf("marshal = %s, want %s", b, want)
	}

	var back struct {
		Class Class `json:"class"`
		Media Media `json:"media"`
	}
	if err := json.Unmarshal([]byte(`{"class":"wan","media":"cellular"}`), &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.Class != ClassWAN || back.Media != MediaCellular {
		t.Fatalf("unmarshal = %v/%v, want wan/cellular", back.Class, back.Media)
	}
}

// The penalty is expressed in the scheduler's own units, and an unmeasured peer
// costs less than a wireless one but more than a wired one.
func TestPenaltyOrdering(t *testing.T) {
	if ClassLocal.Penalty() != 0 || ClassLAN.Penalty() != 0 {
		t.Fatalf("local/lan should be free, got %d/%d", ClassLocal.Penalty(), ClassLAN.Penalty())
	}
	if ClassUnknown.Penalty() <= ClassLAN.Penalty() {
		t.Fatal("an unmeasured peer must not beat a peer measured on the same switch")
	}
	if ClassWAN.Penalty() <= ClassWiFi.Penalty() {
		t.Fatal("wan must cost more than wifi")
	}
}

func TestPathFreshness(t *testing.T) {
	now := time.Date(2026, 3, 1, 12, 0, 0, 0, time.UTC)
	var never Path
	if never.Fresh(now, time.Minute) {
		t.Fatal("a path with no observation is never fresh")
	}
	p := Path{UpdatedAt: now.Add(-30 * time.Second)}
	if !p.Fresh(now, time.Minute) {
		t.Fatal("30s-old reading should be fresh within a minute")
	}
	if p.Fresh(now, 10*time.Second) {
		t.Fatal("30s-old reading should be stale within ten seconds")
	}
}
