// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package linkq

import (
	"context"
	"net"
	"os/exec"
	"time"
)

// hardwarePortTimeout bounds the networksetup call. The command reads a system
// configuration store rather than probing hardware, so it returns in
// milliseconds; the timeout exists so a wedged configd cannot stall a
// classification that has a connection waiting on it.
const hardwarePortTimeout = 3 * time.Second

// networksetupPath is the absolute path to the tool. Absolute because this runs
// inside a supervised service whose PATH is inherited from whatever launched the
// desktop application, and resolving a system utility through an inherited PATH
// is how the wrong binary gets run.
const networksetupPath = "/usr/sbin/networksetup"

// hardwarePorts runs networksetup and returns device name to hardware port name.
// Overridable in tests so the parser can be exercised without the tool.
var hardwarePorts = listHardwarePorts

// interfaceMedia classifies a macOS interface from its hardware port name.
//
// macOS has no ifType equivalent reachable without cgo, and PAIR is a pure-Go
// build with no C toolchain, so the classification comes from the same place the
// Network preference pane gets it: the hardware port list. This is the pattern
// nvpair-node-info already uses for GPU inventory, where ioreg output is parsed
// for facts that are otherwise behind a framework.
//
// The port NAME is what is matched, and it is a display string, so the match is
// deliberately loose and only ever concludes something positive. A port that
// matches nothing is MediaUnknown rather than assumed wired.
func interfaceMedia(name string) Media {
	if iface, err := net.InterfaceByName(name); err == nil && iface.Flags&net.FlagLoopback != 0 {
		return MediaLoopback
	}
	if looksVirtual(name) {
		return MediaVirtual
	}
	ports := hardwarePorts()
	port, ok := ports[name]
	if !ok {
		return MediaUnknown
	}
	return mediaFromPortName(port)
}

// listHardwarePorts parses `networksetup -listallhardwareports`, whose output is
// stanzas of the form:
//
//	Hardware Port: Wi-Fi
//	Device: en0
//	Ethernet Address: 00:00:00:00:00:00
//
// A stanza is emitted when its Device line arrives, so a malformed or truncated
// tail costs only its own entry. A failure to run the tool at all yields an empty
// map, which reads as MediaUnknown for every interface — the correct degradation,
// since the alternative is inventing a classification.
func listHardwarePorts() map[string]string {
	ctx, cancel := context.WithTimeout(context.Background(), hardwarePortTimeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, networksetupPath, "-listallhardwareports").Output()
	if err != nil {
		return nil
	}
	return parseHardwarePorts(string(out))
}
