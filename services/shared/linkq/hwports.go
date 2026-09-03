// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package linkq

// hwports.go holds the macOS hardware-port parsing that media_darwin.go drives.
// It lives in a file with no platform suffix so it compiles and is tested
// everywhere, following nvpair-node-info's ioreg_parse.go: the exec belongs to
// the platform, the parsing of its output does not, and a parser that can only
// run on the machine it targets is a parser nobody runs.

import (
	"bufio"
	"strings"
)

// mediaFromPortName maps a macOS hardware port display name to a Media.
//
// "iPhone USB" is the tether: the cable is USB but the link on the far side is
// cellular, and what matters for placement is the far side. Bluetooth PAN is the
// same shape and is treated the same way.
func mediaFromPortName(port string) Media {
	p := strings.ToLower(port)
	switch {
	case strings.Contains(p, "wi-fi"), strings.Contains(p, "wifi"), strings.Contains(p, "airport"):
		return MediaWireless
	case strings.Contains(p, "iphone"), strings.Contains(p, "ipad"), strings.Contains(p, "bluetooth"):
		return MediaCellular
	case strings.Contains(p, "bridge"), strings.Contains(p, "vpn"):
		return MediaVirtual
	case strings.Contains(p, "ethernet"), strings.Contains(p, "lan"), strings.Contains(p, "thunderbolt"):
		return MediaWired
	default:
		return MediaUnknown
	}
}

// parseHardwarePorts is listHardwarePorts' parser, split out so it can be tested
// against captured output on any platform.
func parseHardwarePorts(out string) map[string]string {
	ports := map[string]string{}
	var current string
	sc := bufio.NewScanner(strings.NewReader(out))
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if v, ok := strings.CutPrefix(line, "Hardware Port:"); ok {
			current = strings.TrimSpace(v)
			continue
		}
		if v, ok := strings.CutPrefix(line, "Device:"); ok {
			dev := strings.TrimSpace(v)
			if dev != "" && current != "" {
				ports[dev] = current
			}
			current = ""
		}
	}
	return ports
}
