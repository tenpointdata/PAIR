// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package peerbook remembers where every paired node has been reachable, so
// finding a peer stops depending on hearing from it.
//
// PAIR learns a peer's address from one place: the multicast announcement it is
// currently making. That works on one friendly LAN and fails in three separate
// ways off it — the announcement is dropped by the access point, or the address
// it carries belongs to a network the listener is not on, or the peer moved and
// the address it announced yesterday now belongs to someone else's laptop.
// Pairing already established durable identity for that peer; only its location
// is in doubt. A book of locations is what closes the gap.
//
// The unit is an Endpoint, not an address, and endpoints carry provenance. Where
// something came from is what makes it possible to rank a claim against
// evidence: a peer's own advertised interface address is a guess about a network
// the listener may not share, while an address that completed a handshake ninety
// seconds ago is a fact. Ranking here is evidence-first for exactly that reason,
// following shared/netpick's finding that address class predicts far less about
// reachability than observation does.
//
// Nothing in this package connects. It says what to try and in what order;
// shared/reach does the trying, and its outcome comes back through RecordSuccess
// and RecordFailure. Keeping the two apart is what lets the ordering be tested
// exhaustively without a network, and what stops a book lookup from blocking on
// a dial.
package peerbook

import (
	"maps"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Kind is where an endpoint came from, which is also most of what is known about
// how far it can be trusted.
type Kind int

const (
	// KindUnknown is an endpoint with no recorded provenance. It is ranked last
	// among untried candidates rather than discarded: it is still an address
	// somebody recorded, and the cost of trying one is a TCP handshake.
	KindUnknown Kind = iota
	// KindLocal is an interface address the peer published for itself — today's
	// "ips=" list. Authoritative about the peer's own hardware, and silent about
	// whether the listener shares a network with it.
	KindLocal
	// KindName is a DNS name a user configured for the node, dynamic DNS
	// included. The only endpoint kind that survives renumbering, which makes it
	// the one worth asking a user for.
	KindName
	// KindMapped is an inbound path a user deliberately opened: a static
	// port-forward, or a UPnP-IGD or NAT-PMP lease they opted into. PAIR never
	// creates one on its own.
	KindMapped
	// KindReflexive is the address a peer observed this node connecting FROM,
	// reported back over the authenticated channel. It is what a STUN server
	// would provide, obtained without one, because every cluster peer already
	// terminates our connections and can see the same thing.
	KindReflexive
	// KindRelay is not an address for the peer at all: it is another member that
	// has a path to it. Ranked apart from the rest rather than below them,
	// because using one spends a third machine's bandwidth and is a different
	// decision from choosing between two ways to dial directly.
	KindRelay
)

// String returns the wire and log spelling of a kind.
func (k Kind) String() string {
	switch k {
	case KindLocal:
		return "local"
	case KindName:
		return "name"
	case KindMapped:
		return "mapped"
	case KindReflexive:
		return "reflexive"
	case KindRelay:
		return "relay"
	default:
		return "unknown"
	}
}

// ParseKind is String's inverse. An unrecognized spelling reads as KindUnknown
// rather than failing the record: a book written by a future build should lose
// the meaning of one field, not the address it describes.
func ParseKind(s string) Kind {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "local":
		return KindLocal
	case "name":
		return KindName
	case "mapped":
		return KindMapped
	case "reflexive":
		return KindReflexive
	case "relay":
		return KindRelay
	default:
		return KindUnknown
	}
}

// MarshalText renders a Kind as its spelling, so the on-disk book stays readable
// and adding a kind later does not renumber the ones already written.
func (k Kind) MarshalText() ([]byte, error) { return []byte(k.String()), nil }

// UnmarshalText accepts the spelling, tolerating an unknown one.
func (k *Kind) UnmarshalText(b []byte) error { *k = ParseKind(string(b)); return nil }

// kindTiebreak orders kinds when nothing else separates two untried endpoints.
// Lower is tried first. It is only ever a tiebreak: evidence beats it whenever
// there is any.
//
// Local leads because on a shared LAN it is both the fastest path and the most
// common case. Name follows because it is the kind that survives a roam. Mapped
// and reflexive are inbound paths through NAT, which work less often than either.
// Relay is last by construction.
func (k Kind) kindTiebreak() int {
	switch k {
	case KindLocal:
		return 0
	case KindName:
		return 1
	case KindMapped:
		return 2
	case KindReflexive:
		return 3
	case KindRelay:
		return 4
	default:
		return 5
	}
}

// Endpoint is one recorded way to reach a peer, with the evidence for and
// against it.
type Endpoint struct {
	// Kind is the provenance.
	Kind Kind `json:"kind"`
	// Host is an IP literal or DNS name, never a host:port pair. Ports belong to
	// services, and one endpoint reaches all of a peer's services.
	Host string `json:"host"`
	// Ports overrides the peer's advertised service ports for this endpoint,
	// keyed by the compact service key the discovery record uses ("ni", "cl",
	// "wl", …). It is populated for endpoints that traverse address translation,
	// where the port a peer listens on is not the port the world connects to.
	// Empty means the peer's own advertised ports apply.
	Ports map[string]int `json:"ports,omitempty"`
	// Via is the node UUID of the relaying member, for KindRelay only.
	Via string `json:"via,omitempty"`
	// Source records who told us: "mdns", "manual", "observed", or
	// "gossip:<uuid>". Kept because a bad endpoint is worth tracing to whoever
	// keeps advertising it.
	Source string `json:"source,omitempty"`
	// FirstSeen is when this endpoint entered the book.
	FirstSeen time.Time `json:"firstSeen"`
	// LastAdvertised is when it was most recently claimed by anyone.
	LastAdvertised time.Time `json:"lastAdvertised,omitempty"`
	// LastSuccess is when a connection to it last completed. The single most
	// important field in the struct: it is the only one that is evidence rather
	// than a claim.
	LastSuccess time.Time `json:"lastSuccess,omitempty"`
	// LastFailure is when an attempt last failed.
	LastFailure time.Time `json:"lastFailure,omitempty"`
	// Failures counts consecutive failures since the last success. Reset by one
	// success, because a single answer proves the address works now and the
	// history was about a network the peer has since left.
	Failures int `json:"failures,omitempty"`
	// Successes counts lifetime successes, for diagnostics.
	Successes int `json:"successes,omitempty"`
}

// key identifies an endpoint for merge purposes. Ports deliberately do not
// participate: a NAT rebinding that moves the external port is the same endpoint
// with new information, and treating it as a different one would accumulate a
// row per rebinding until the cap evicted the useful history along with the rest.
func (e Endpoint) key() string { return e.Kind.String() + "|" + e.Host + "|" + e.Via }

// Peer is one paired node's entry: its identity, the ports it advertises, and
// every recorded way to reach it.
type Peer struct {
	// NodeUUID is the peer's stable identity — the same one its pinned
	// certificate is filed under. It, not an address, is what the book is keyed
	// by, which is the entire premise: identity is durable and location is not.
	NodeUUID string `json:"nodeUuid"`
	// Name is the peer's friendly name, for display only.
	Name string `json:"name,omitempty"`
	// Ports are the service ports the peer advertises, keyed by compact service
	// key. An endpoint with no override of its own uses these.
	Ports map[string]int `json:"ports,omitempty"`
	// Endpoints is every recorded way to reach it, in no particular order:
	// ranking is computed on read, because it depends on the current time.
	Endpoints []Endpoint `json:"endpoints,omitempty"`
	// UpdatedAt is when this entry last changed.
	UpdatedAt time.Time `json:"updatedAt"`
}

// clone returns a deep copy, so a snapshot handed to a caller cannot be mutated
// into the book.
func (p Peer) clone() Peer {
	out := p
	out.Ports = maps.Clone(p.Ports)
	out.Endpoints = make([]Endpoint, len(p.Endpoints))
	for i, e := range p.Endpoints {
		e.Ports = maps.Clone(e.Ports)
		out.Endpoints[i] = e
	}
	return out
}

// Ranking thresholds. These are about how long evidence stays evidence, which is
// a different question for a success than for a claim.
const (
	// freshSuccess is how recently a handshake must have completed for an
	// endpoint to be tried first without further thought. It is short because
	// the thing that invalidates it — a roam, a sleep, a DHCP change — happens in
	// seconds and gives no notice.
	freshSuccess = 3 * time.Minute
	// provenSuccess is how long a past success keeps an endpoint ahead of one
	// that has never been tried. A day, because the addresses a machine uses
	// repeat: the same office LAN, the same home Wi-Fi, the same tether. An
	// address that worked yesterday is a far better first guess than one nobody
	// has attempted, even after a night away.
	provenSuccess = 24 * time.Hour
	// failureDemotion is how many consecutive failures drop an endpoint below
	// untried candidates. Three, so a peer that is merely rebooting is not
	// reordered out of its own best address, while one that has genuinely moved
	// stops costing a dial timeout on every attempt.
	failureDemotion = 3
)

// Limits on how much a book keeps. Both exist to bound work rather than storage:
// candidates are probed concurrently, so an unbounded list is an unbounded burst
// of connections aimed at one peer.
const (
	// MaxEndpointsPerPeer caps recorded endpoints. Generous enough for a
	// multi-homed machine seen on several networks, small enough that the eviction
	// below never has to think hard.
	MaxEndpointsPerPeer = 16
	// MaxCandidates caps one Candidates call. Beyond this, a probe sweep costs
	// more than the failover it is trying to accelerate.
	MaxCandidates = 8
	// staleEndpoint is how long an endpoint survives with neither an
	// advertisement nor a success before pruning removes it.
	staleEndpoint = 30 * 24 * time.Hour
)

// rank is an endpoint's sort tier at time now; lower is tried first.
//
// Tiers rather than a weighted score, deliberately. A score invites tuning
// constants against a network nobody has, and makes "why did it try that one
// first" unanswerable. Four tiers can be read off an entry and explained in a
// log line.
func (e Endpoint) rank(now time.Time) int {
	switch {
	case !e.LastSuccess.IsZero() && now.Sub(e.LastSuccess) <= freshSuccess && e.Failures == 0:
		return 0 // known good, right now
	case !e.LastSuccess.IsZero() && now.Sub(e.LastSuccess) <= provenSuccess && e.Failures < failureDemotion:
		return 1 // worked recently enough to be worth trying first
	case e.Failures >= failureDemotion:
		return 3 // repeatedly refused; try it, but after everything else
	default:
		return 2 // never tried, or tried long ago
	}
}

// lastEvidence is the most recent moment anything at all was learned about an
// endpoint, whether by reaching it or by someone mentioning it.
func (e Endpoint) lastEvidence() time.Time {
	last := e.LastAdvertised
	if e.LastSuccess.After(last) {
		last = e.LastSuccess
	}
	if last.IsZero() {
		last = e.FirstSeen
	}
	return last
}

// less orders two endpoints for the candidate list. Total and deterministic: two
// runs of the same book produce the same order, which is what makes a failover
// reproducible in a bug report.
//
// Which recency decides a tie depends on which tier the tie is in, and the
// distinction is the interesting part. Among endpoints that have a usable
// success behind them, the more recent SUCCESS leads: a handshake we completed
// outranks a claim someone made, however fresh the claim. Among endpoints
// without one, the more recent EVIDENCE of any kind leads, because a peer
// announcing an address right now is telling us more than a success from before
// it moved.
func less(a, b Endpoint, now time.Time) bool {
	ra, rb := a.rank(now), b.rank(now)
	if ra != rb {
		return ra < rb
	}
	if ra <= 1 {
		if !a.LastSuccess.Equal(b.LastSuccess) {
			return a.LastSuccess.After(b.LastSuccess)
		}
	} else if ea, eb := a.lastEvidence(), b.lastEvidence(); !ea.Equal(eb) {
		return ea.After(eb)
	}
	if a.Failures != b.Failures {
		return a.Failures < b.Failures
	}
	if ka, kb := a.Kind.kindTiebreak(), b.Kind.kindTiebreak(); ka != kb {
		return ka < kb
	}
	return a.Host < b.Host
}

// sortEndpoints orders a copy of eps for candidate selection.
func sortEndpoints(eps []Endpoint, now time.Time) []Endpoint {
	out := make([]Endpoint, len(eps))
	copy(out, eps)
	sort.SliceStable(out, func(i, j int) bool { return less(out[i], out[j], now) })
	return out
}

// address renders host:port for a service, using the endpoint's own port
// override when it has one and the peer's advertised port otherwise. ok is false
// when neither supplies a port, which is not an error: a peer that has never
// advertised, say, an engine-control port simply does not offer that service,
// and inventing a default would produce a candidate that can only time out.
func (e Endpoint) address(service string, peerPorts map[string]int) (string, bool) {
	port, ok := e.Ports[service]
	if !ok {
		port, ok = peerPorts[service]
	}
	if !ok || port <= 0 || port > 65535 || e.Host == "" {
		return "", false
	}
	return joinHostPort(e.Host, port), true
}

// joinHostPort brackets an IPv6 literal and leaves anything else alone. It does
// not use net.JoinHostPort because a DNS name must not be bracketed and that
// function decides by looking for a colon, which a name never has but which this
// keeps explicit.
func joinHostPort(host string, port int) string {
	if strings.Contains(host, ":") && !strings.HasPrefix(host, "[") {
		return "[" + host + "]:" + strconv.Itoa(port)
	}
	return host + ":" + strconv.Itoa(port)
}
