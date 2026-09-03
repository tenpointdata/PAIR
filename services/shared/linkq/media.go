// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package linkq

import (
	"net"
	"strings"
	"sync"
	"time"
)

// mediaCacheTTL bounds how long a per-interface media answer is reused.
//
// The answer changes only when hardware does — a dock is attached, a USB adapter
// is unplugged, an interface is renamed — so a long TTL is correct on the
// merits. It is bounded at all because the alternative on macOS is running
// networksetup on every classification, and a classification happens per
// connection. Callers watching shared/netmon should call ResetMediaCache on a
// network change rather than waiting this out.
const mediaCacheTTL = 5 * time.Minute

// egressProbeTimeout bounds the routing-table lookup in localMediaTo. A UDP
// "connection" sends nothing, so this only ever expires when the host has no
// route at all, which is a real answer worth returning quickly.
const egressProbeTimeout = 250 * time.Millisecond

var (
	mediaMu    sync.Mutex
	mediaCache = map[string]mediaEntry{}
)

type mediaEntry struct {
	media Media
	at    time.Time
}

// InterfaceMedia returns what kind of link an interface is, by name, using the
// platform detector in media_<goos>.go. Results are cached for mediaCacheTTL.
//
// A name that does not resolve, or a platform with no detector, yields
// MediaUnknown. That is the honest answer and consumers are built to accept it:
// Classify lets topology stand when media is unknown, and anything that needs
// certainty reads Path.MediaKnown instead.
func InterfaceMedia(name string) Media {
	if name == "" {
		return MediaUnknown
	}
	now := time.Now()
	mediaMu.Lock()
	if e, ok := mediaCache[name]; ok && now.Sub(e.at) < mediaCacheTTL {
		mediaMu.Unlock()
		return e.media
	}
	mediaMu.Unlock()

	m := interfaceMedia(name)

	mediaMu.Lock()
	mediaCache[name] = mediaEntry{media: m, at: now}
	mediaMu.Unlock()
	return m
}

// ResetMediaCache discards every cached media answer. Call it when the host's
// network configuration changes — shared/netmon exists to say when — so a
// machine that just moved from Wi-Fi to a dock is classified as docked without
// waiting out the TTL.
func ResetMediaCache() {
	mediaMu.Lock()
	clear(mediaCache)
	mediaMu.Unlock()
}

// PathTo reports how this host would reach remote and what local interface would
// carry it: the two inputs Classify needs, gathered together because gathering
// them separately means enumerating interfaces twice for one answer.
//
// It resolves the topology from this host's own addresses, so it distinguishes a
// peer on one of our subnets from one behind a router. It cannot see a relay —
// that is the caller's own decision, made when no direct endpoint answered — so
// a relayed connection passes TopologyRelayed to Classify directly rather than
// asking here.
func PathTo(remote net.IP) (Topology, Media) {
	if remote == nil {
		return TopologyUnknown, MediaUnknown
	}
	if remote.IsLoopback() {
		return TopologyLoopback, MediaLoopback
	}
	ifaces, err := net.Interfaces()
	if err != nil {
		return TopologyUnknown, MediaUnknown
	}
	for _, iface := range ifaces {
		if iface.Flags&net.FlagUp == 0 {
			continue
		}
		addrs, err := iface.Addrs()
		if err != nil {
			continue
		}
		for _, a := range addrs {
			ipnet, ok := a.(*net.IPNet)
			if !ok || ipnet.IP.IsLoopback() {
				continue
			}
			if ipnet.Contains(remote) {
				return TopologySameSubnet, InterfaceMedia(iface.Name)
			}
		}
	}
	return TopologyRouted, egressMedia(remote)
}

// egressMedia reports the media of the interface the host's routing table would
// use to reach remote.
//
// It asks the kernel by "connecting" a UDP socket, which selects a route and
// binds a local address without putting a single packet on the wire. Reading the
// answer from the routing table is the only way to get this right on a
// multi-homed host: the destination is by definition not on any of our subnets,
// so no address comparison can identify which interface carries it, and guessing
// the default route misclassifies every host with a VPN or a second uplink.
func egressMedia(remote net.IP) Media {
	d := net.Dialer{Timeout: egressProbeTimeout}
	conn, err := d.Dial("udp", net.JoinHostPort(remote.String(), "9"))
	if err != nil {
		return MediaUnknown
	}
	defer func() { _ = conn.Close() }()
	local, ok := conn.LocalAddr().(*net.UDPAddr)
	if !ok || local.IP == nil {
		return MediaUnknown
	}
	name, ok := interfaceNameForIP(local.IP)
	if !ok {
		return MediaUnknown
	}
	return InterfaceMedia(name)
}

// interfaceNameForIP finds which interface holds ip.
func interfaceNameForIP(ip net.IP) (string, bool) {
	ifaces, err := net.Interfaces()
	if err != nil {
		return "", false
	}
	for _, iface := range ifaces {
		addrs, err := iface.Addrs()
		if err != nil {
			continue
		}
		for _, a := range addrs {
			ipnet, ok := a.(*net.IPNet)
			if ok && ipnet.IP.Equal(ip) {
				return iface.Name, true
			}
		}
	}
	return "", false
}

// virtualNamePrefixes are interface-name prefixes that identify a software
// interface on every platform that uses them. Name matching is a weak signal and
// is only ever used to CONFIRM virtual, never to confirm physical: a real
// adapter can be renamed to anything, but nothing renames itself to docker0.
var virtualNamePrefixes = []string{
	"docker", "br-", "veth", "virbr", "vmnet", "vboxnet",
	"tun", "tap", "wg", "utun", "zt", "tailscale", "ham",
}

// looksVirtual reports whether an interface name is one of the well-known
// software interfaces. Used by the platform detectors as a last check before
// concluding "wired", so a container bridge is not offered to the scheduler as
// though it were a cable.
func looksVirtual(name string) bool {
	n := strings.ToLower(name)
	for _, p := range virtualNamePrefixes {
		if strings.HasPrefix(n, p) {
			return true
		}
	}
	return false
}

// cellularNamePrefixes identify mobile-broadband interfaces by name. Unlike
// virtual detection this is the primary signal on Linux, where a WWAN modem
// commonly presents as an ordinary Ethernet-typed device and the only thing
// distinguishing it is the driver's naming.
var cellularNamePrefixes = []string{"wwan", "wwp", "rmnet", "ppp", "cdc-wdm"}

// looksCellular reports whether an interface name is a mobile-broadband one.
func looksCellular(name string) bool {
	n := strings.ToLower(name)
	for _, p := range cellularNamePrefixes {
		if strings.HasPrefix(n, p) {
			return true
		}
	}
	return false
}
