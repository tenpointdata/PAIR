// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package linkq

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// sysClassNet is the kernel's interface directory. A variable so tests can point
// the detector at a fixture tree instead of the running machine's hardware.
var sysClassNet = "/sys/class/net"

// ARPHRD values from the kernel's if_arp.h that identify a link type on sight.
// Ethernet is not among them on purpose: 802.11 interfaces report ARPHRD_ETHER
// too, because they present an Ethernet-shaped device to the stack. Telling them
// apart needs the wireless attributes below, which is why type alone is only
// consulted for the cases where it is decisive.
const (
	arphrdLoopback = 772
	arphrdPPP      = 512
	arphrdRawIP    = 519 // rmnet and similar modem interfaces
)

// interfaceMedia classifies a Linux interface from sysfs.
//
// The order matters and is not arbitrary. Loopback and the modem link types are
// decisive on their own. Wireless is identified by the attributes the cfg80211
// stack adds — a phy80211 symlink, or the legacy wireless directory — rather
// than by device type, which reports Ethernet for Wi-Fi. Only then does the
// absence of a device symlink mark an interface virtual: veth, bridges, tunnels,
// and VPNs have no backing device, while every real adapter does. Anything left
// is a cable.
func interfaceMedia(name string) Media {
	base := filepath.Join(sysClassNet, name)
	if _, err := os.Stat(base); err != nil {
		return MediaUnknown
	}

	switch readSysInt(filepath.Join(base, "type")) {
	case arphrdLoopback:
		return MediaLoopback
	case arphrdPPP, arphrdRawIP:
		return MediaCellular
	}

	if isDir(filepath.Join(base, "phy80211")) || isDir(filepath.Join(base, "wireless")) {
		return MediaWireless
	}
	if looksCellular(name) {
		return MediaCellular
	}
	if _, err := os.Stat(filepath.Join(base, "device")); err != nil {
		return MediaVirtual
	}
	if looksVirtual(name) {
		return MediaVirtual
	}
	return MediaWired
}

// readSysInt reads a single-integer sysfs attribute, returning -1 when it is
// missing or unparseable. Sysfs reads fail routinely on interfaces that are
// being torn down, so an unreadable attribute is an ordinary outcome and not
// worth logging.
func readSysInt(path string) int {
	b, err := os.ReadFile(path)
	if err != nil {
		return -1
	}
	v, err := strconv.Atoi(strings.TrimSpace(string(b)))
	if err != nil {
		return -1
	}
	return v
}

// isDir reports whether path exists and is a directory. A symlink to one counts:
// phy80211 is always a symlink.
func isDir(path string) bool {
	fi, err := os.Stat(path)
	return err == nil && fi.IsDir()
}
