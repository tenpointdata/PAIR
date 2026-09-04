// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package linkq answers one question every scheduling and placement decision in
// a cluster wider than a single switch needs, and none of them can currently
// ask: what kind of link is this peer on, and how good is it right now.
//
// PAIR was built for a LAN, where the answer is the same for every peer and can
// safely be assumed. That assumption is what breaks first when a cluster spans
// Wi-Fi networks. nvpair-job-scheduler ranks nodes by pending work and GPU
// pressure with nothing in its input distinguishing a peer across a 10 GbE
// switch from one across a congested 2.4 GHz link, so a wireless node with an
// idle GPU outranks a wired node with a busy one — correct arithmetic, wrong
// answer. For distributed inference the same gap stops being a performance
// question and becomes a correctness one: a pool member on a flapping link
// takes the entire pool down with it when it drops.
//
// Two independent inputs produce the answer, and keeping them separate is the
// design:
//
//   - Media is what the operating system says a local interface IS — wired,
//     wireless, cellular, virtual. Only the host itself can see this, so each
//     node classifies its own interfaces and publishes the result; a peer cannot
//     derive it by observation.
//   - Path quality is what the link DOES — round-trip time, throughput, and
//     whether it stays up. This is measured, not declared, and it is measured
//     from traffic that was going to happen anyway (see Tracker).
//
// Nothing here probes. A synthetic bandwidth test on a metered phone tether is
// a cost the user did not agree to, and a periodic one is a cost they pay
// forever. Every sample this package smooths comes from a request some other
// part of PAIR had already decided to send.
//
// The classification is deliberately coarse, in the same spirit as the
// scheduler's 0-3 GPU pressure. Five classes carry enough signal to rank a peer,
// gate a pool donor, and explain a decision to a user, and none of them invites
// a consumer to believe a precision that a smoothed EWMA over opportunistic
// samples does not have.
package linkq

import (
	"fmt"
	"strings"
	"time"
)

// Media is what an operating system reports a network interface to be. It is a
// property of one host's own interface, never of a path: a wired node talking to
// a wireless one has a wireless path and a wired interface, and both facts are
// needed to classify the link between them.
type Media int

const (
	// MediaUnknown is the honest answer on a platform or interface where the
	// media type could not be read. It is never treated as good news: consumers
	// that need certainty (the pool donor gate) require a positively-known
	// wired media rather than the absence of a wireless one.
	MediaUnknown Media = iota
	// MediaLoopback is the host itself.
	MediaLoopback
	// MediaWired is Ethernet and anything else that reaches the network over a
	// cable.
	MediaWired
	// MediaWireless is 802.11 in any of its generations. PAIR does not
	// distinguish them: the band, the standard, and the negotiated rate say much
	// less about the link a peer will actually get than the measured round-trip
	// time does, and the measurement is already being taken.
	MediaWireless
	// MediaCellular is a mobile broadband interface or a phone tether. It is
	// separated from MediaWireless because the difference that matters is not
	// the radio, it is that the link is metered, carrier-NATted, and priced.
	MediaCellular
	// MediaVirtual is a bridge, tunnel, container, or VPN interface — anything
	// whose quality is a property of the interface it rides on rather than of
	// itself. It cannot be classified further from the host's own vantage point.
	MediaVirtual
)

// String returns the wire and log spelling of a media type.
func (m Media) String() string {
	switch m {
	case MediaLoopback:
		return "loopback"
	case MediaWired:
		return "wired"
	case MediaWireless:
		return "wireless"
	case MediaCellular:
		return "cellular"
	case MediaVirtual:
		return "virtual"
	default:
		return "unknown"
	}
}

// ParseMedia is String's inverse, for reading a peer's published classification.
// An unrecognized spelling reads as MediaUnknown rather than failing: a future
// media type this build has never heard of is exactly as informative as no
// answer, and refusing the whole record over one field would discard the rest of
// a peer's usable state.
func ParseMedia(s string) Media {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "loopback":
		return MediaLoopback
	case "wired":
		return MediaWired
	case "wireless":
		return MediaWireless
	case "cellular":
		return MediaCellular
	case "virtual":
		return MediaVirtual
	default:
		return MediaUnknown
	}
}

// MarshalText renders a Media as its string spelling so a JSON payload carries
// "wireless" rather than 3. Wire formats outlive the constant order they were
// written against, and a numeric enum makes inserting a value a breaking change.
func (m Media) MarshalText() ([]byte, error) { return []byte(m.String()), nil }

// UnmarshalText accepts the string spelling, tolerating an unknown one.
func (m *Media) UnmarshalText(b []byte) error { *m = ParseMedia(string(b)); return nil }

// Class is the quality band of a path between this node and one peer. It is
// derived from topology and both ends' media (see Classify), never declared.
type Class int

const (
	// ClassUnknown means nothing has been observed about this peer yet. It is
	// distinct from every other class and must not be substituted for one: a
	// consumer that treats it as ClassLAN has invented evidence, and one that
	// treats it as ClassWAN penalizes a node for having been quiet.
	ClassUnknown Class = iota
	// ClassLocal is this node itself, over loopback.
	ClassLocal
	// ClassLAN is a directly-reachable peer on the same subnet over wired media
	// at both ends: the link PAIR was designed around.
	ClassLAN
	// ClassWiFi is a directly-reachable peer with 802.11 somewhere on the path.
	// Usable for routed inference, and the first class a pool donor gate should
	// look at twice.
	ClassWiFi
	// ClassWAN is a peer that is routed, relayed, or reached over a cellular
	// link — anything that is not on this segment. Fine for control traffic and
	// for a routed request whose response streams back; wrong for anything that
	// moves gigabytes or serializes a round trip per token.
	ClassWAN
)

// String returns the wire and log spelling of a class.
func (c Class) String() string {
	switch c {
	case ClassLocal:
		return "local"
	case ClassLAN:
		return "lan"
	case ClassWiFi:
		return "wifi"
	case ClassWAN:
		return "wan"
	default:
		return "unknown"
	}
}

// ParseClass is String's inverse. An unrecognized spelling reads as
// ClassUnknown, for the reason ParseMedia gives.
func ParseClass(s string) Class {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "local":
		return ClassLocal
	case "lan":
		return ClassLAN
	case "wifi":
		return ClassWiFi
	case "wan":
		return ClassWAN
	default:
		return ClassUnknown
	}
}

// MarshalText renders a Class as its string spelling.
func (c Class) MarshalText() ([]byte, error) { return []byte(c.String()), nil }

// UnmarshalText accepts the string spelling, tolerating an unknown one.
func (c *Class) UnmarshalText(b []byte) error { *c = ParseClass(string(b)); return nil }

// Topology is how a peer is reached, independent of what either end's interfaces
// are made of. It is the base the media types then demote.
type Topology int

const (
	// TopologyUnknown means the path has not been established yet.
	TopologyUnknown Topology = iota
	// TopologyLoopback is this node.
	TopologyLoopback
	// TopologySameSubnet is a direct connection to an address on one of this
	// host's own subnets — no router in between.
	TopologySameSubnet
	// TopologyRouted is a direct connection to an address that is not on any of
	// this host's subnets. At least one router forwards it.
	TopologyRouted
	// TopologyRelayed is a connection carried by another cluster member because
	// no direct endpoint answered. It is the slowest path by construction: two
	// hops, and the second one is whatever the relay itself has.
	TopologyRelayed
)

// Classify derives a path's class from how the peer is reached and what both
// ends' interfaces are.
//
// Topology sets the base and media may only make it worse. That direction is the
// whole rule, and it is what makes an unmeasurable media type safe: on a
// platform where the local media cannot be read, the topology answer stands
// rather than every peer collapsing to ClassUnknown, while a consumer that
// genuinely needs certainty asks Path.MediaKnown instead of reading optimism
// into the class.
//
// Cellular demotes all the way to ClassWAN rather than to ClassWiFi even on the
// same subnet, because what distinguishes it is not the radio: a tethered link
// is metered and carrier-NATted, and moving a model's weights across one is a
// bill as well as a delay.
func Classify(topology Topology, local, remote Media) Class {
	if topology == TopologyLoopback || local == MediaLoopback || remote == MediaLoopback {
		return ClassLocal
	}
	base := ClassUnknown
	switch topology {
	case TopologySameSubnet:
		base = ClassLAN
	case TopologyRouted, TopologyRelayed:
		base = ClassWAN
	case TopologyUnknown:
		return ClassUnknown
	}
	if local == MediaCellular || remote == MediaCellular {
		return ClassWAN
	}
	if (local == MediaWireless || remote == MediaWireless) && base == ClassLAN {
		return ClassWiFi
	}
	return base
}

// Path is everything known about the link to one peer: how it is classified, how
// it is performing, and how much that has been changing.
//
// Every duration and rate here is smoothed over opportunistic samples, so treat
// them as a band rather than a measurement. Samples says how much evidence is
// behind them and UpdatedAt says how old it is; a consumer that acts on RTT
// without checking both is acting on a number that may have been taken once, on
// a different network, an hour ago.
type Path struct {
	// Class is the current quality band.
	Class Class `json:"class"`
	// LocalMedia and RemoteMedia are the two ends' interface types, carried so a
	// consumer can explain a classification rather than only apply it.
	LocalMedia  Media `json:"localMedia"`
	RemoteMedia Media `json:"remoteMedia"`
	// MediaKnown reports whether BOTH ends' media were positively identified.
	// The pool donor gate requires this: "not known to be wireless" is not the
	// same claim as "known to be wired", and lending a machine's VRAM across a
	// link on the strength of the first one is how a pool acquires a member that
	// drops it.
	MediaKnown bool `json:"mediaKnown"`
	// Topology is how the peer is currently reached.
	Topology Topology `json:"-"`
	// RTT is the smoothed round-trip time, or zero when none has been observed.
	// It is the term that dominates token generation across a pool: the bytes
	// that cross a layer boundary per token are trivial, and the serialized hop
	// is not.
	RTT time.Duration `json:"rttMs"`
	// Throughput is the smoothed observed rate in bytes per second, or zero when
	// no sample large enough to be meaningful has been seen. It is the term that
	// dominates model load and prompt processing.
	Throughput uint64 `json:"throughputBps"`
	// Stability is an exponentially-weighted fraction of reachability
	// observations that succeeded, in [0,1]. It forgives slowly on purpose: a
	// donor that dropped a pool once is not vindicated by the next successful
	// probe thirty seconds later.
	Stability float64 `json:"stability"`
	// ClassChanges counts how many times this peer's class has changed since
	// tracking began. A roaming laptop reveals itself here long before any single
	// reading looks wrong.
	ClassChanges int `json:"classChanges"`
	// Samples is how many quality observations of any kind are behind these
	// numbers.
	Samples int `json:"samples"`
	// UpdatedAt is when the most recent observation landed.
	UpdatedAt time.Time `json:"updatedAt"`
}

// Fresh reports whether the path was observed within ttl of now. Stale readings
// are not wrong so much as unowned: they describe the network the peer was on,
// which after a roam is a different network.
func (p Path) Fresh(now time.Time, ttl time.Duration) bool {
	return !p.UpdatedAt.IsZero() && now.Sub(p.UpdatedAt) <= ttl
}

// String renders a path for a log line: class, what is known about it, and how
// much evidence there is.
func (p Path) String() string {
	var b strings.Builder
	b.WriteString(p.Class.String())
	if p.RTT > 0 {
		fmt.Fprintf(&b, " rtt=%s", p.RTT.Round(time.Millisecond))
	}
	if p.Throughput > 0 {
		fmt.Fprintf(&b, " bps=%d", p.Throughput)
	}
	fmt.Fprintf(&b, " stability=%.2f samples=%d", p.Stability, p.Samples)
	if p.ClassChanges > 0 {
		fmt.Fprintf(&b, " changes=%d", p.ClassChanges)
	}
	return b.String()
}
