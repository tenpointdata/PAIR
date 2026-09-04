// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package linkq

import (
	"net"
	"testing"
)

func TestLooksVirtualRecognizesSoftwareInterfaces(t *testing.T) {
	for _, name := range []string{"docker0", "br-1a2b3c", "veth9f2", "tun0", "wg0", "utun3", "vEthernet (Default Switch)"} {
		if !looksVirtual(name) {
			t.Errorf("%q should look virtual", name)
		}
	}
	for _, name := range []string{"eth0", "eno1", "enp3s0", "en0", "wlan0", "Ethernet", "Wi-Fi"} {
		if looksVirtual(name) {
			t.Errorf("%q should not look virtual", name)
		}
	}
}

func TestLooksCellularRecognizesModems(t *testing.T) {
	for _, name := range []string{"wwan0", "wwp0s20f0u6", "rmnet_data0", "ppp0"} {
		if !looksCellular(name) {
			t.Errorf("%q should look cellular", name)
		}
	}
	for _, name := range []string{"wlan0", "eth0", "en0"} {
		if looksCellular(name) {
			t.Errorf("%q should not look cellular", name)
		}
	}
}

func TestInterfaceMediaOnAnEmptyNameIsUnknown(t *testing.T) {
	if got := InterfaceMedia(""); got != MediaUnknown {
		t.Fatalf("InterfaceMedia(\"\") = %v, want %v", got, MediaUnknown)
	}
}

// PathTo must recognize this host before doing any interface work: loopback is
// the one answer that is the same on every platform and in every test
// environment, including a container with no usable interfaces at all.
func TestPathToLoopback(t *testing.T) {
	topo, media := PathTo(net.ParseIP("127.0.0.1"))
	if topo != TopologyLoopback || media != MediaLoopback {
		t.Fatalf("PathTo(127.0.0.1) = %v/%v, want loopback/loopback", topo, media)
	}
	if got := Classify(topo, media, MediaUnknown); got != ClassLocal {
		t.Fatalf("classified loopback as %v", got)
	}
}

func TestPathToNilAddressIsUnknown(t *testing.T) {
	topo, media := PathTo(nil)
	if topo != TopologyUnknown || media != MediaUnknown {
		t.Fatalf("PathTo(nil) = %v/%v, want unknown/unknown", topo, media)
	}
}

// The cache exists so a macOS classification does not exec networksetup per
// connection. Its correctness requirement is only that resetting it is
// observable, which is what a network change depends on.
func TestResetMediaCacheClearsEntries(t *testing.T) {
	InterfaceMedia("lo")
	mediaMu.Lock()
	cached := len(mediaCache)
	mediaMu.Unlock()
	if cached == 0 {
		t.Fatal("a classification should populate the cache")
	}

	ResetMediaCache()
	mediaMu.Lock()
	defer mediaMu.Unlock()
	if len(mediaCache) != 0 {
		t.Fatalf("cache still holds %d entries after reset", len(mediaCache))
	}
}

func TestParseHardwarePorts(t *testing.T) {
	const out = `Hardware Port: Wi-Fi
Device: en0
Ethernet Address: 3c:22:fb:00:00:01

Hardware Port: Thunderbolt Ethernet Slot 1
Device: en5
Ethernet Address: 3c:22:fb:00:00:02

Hardware Port: iPhone USB
Device: en8
Ethernet Address: N/A

Hardware Port: Thunderbolt Bridge
Device: bridge0
Ethernet Address: N/A

VLAN Configurations
===================
`
	got := parseHardwarePorts(out)
	want := map[string]string{
		"en0":     "Wi-Fi",
		"en5":     "Thunderbolt Ethernet Slot 1",
		"en8":     "iPhone USB",
		"bridge0": "Thunderbolt Bridge",
	}
	if len(got) != len(want) {
		t.Fatalf("parsed %d ports, want %d: %v", len(got), len(want), got)
	}
	for dev, port := range want {
		if got[dev] != port {
			t.Errorf("device %s = %q, want %q", dev, got[dev], port)
		}
	}
}

// A truncated stanza costs only its own entry — the tool's output is read
// incrementally and a missing Device line simply never emits.
func TestParseHardwarePortsToleratesTruncation(t *testing.T) {
	const out = `Hardware Port: Wi-Fi
Device: en0

Hardware Port: Ethernet
`
	got := parseHardwarePorts(out)
	if len(got) != 1 || got["en0"] != "Wi-Fi" {
		t.Fatalf("parsed %v, want only en0", got)
	}
}

func TestMediaFromPortName(t *testing.T) {
	cases := map[string]Media{
		"Wi-Fi":                       MediaWireless,
		"AirPort":                     MediaWireless,
		"Thunderbolt Ethernet Slot 1": MediaWired,
		"USB 10/100/1000 LAN":         MediaWired,
		"iPhone USB":                  MediaCellular,
		"Bluetooth PAN":               MediaCellular,
		"Thunderbolt Bridge":          MediaVirtual,
		"Something Else Entirely":     MediaUnknown,
	}
	for port, want := range cases {
		if got := mediaFromPortName(port); got != want {
			t.Errorf("mediaFromPortName(%q) = %v, want %v", port, got, want)
		}
	}
}
