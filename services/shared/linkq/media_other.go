// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows && !linux && !darwin

package linkq

import "net"

// interfaceMedia is the fallback for platforms with no media detector. It
// reports what can be known from the standard library alone — loopback, and the
// software interfaces that name themselves — and MediaUnknown for everything
// else.
//
// Returning MediaUnknown rather than guessing MediaWired is the point. Classify
// lets topology stand when media is unknown, so an unclassified platform still
// ranks its peers sensibly; guessing wired would instead tell a pool planner
// that a laptop on hotel Wi-Fi is a safe place to put twelve gigabytes of model
// weights.
func interfaceMedia(name string) Media {
	if iface, err := net.InterfaceByName(name); err == nil && iface.Flags&net.FlagLoopback != 0 {
		return MediaLoopback
	}
	if looksVirtual(name) {
		return MediaVirtual
	}
	if looksCellular(name) {
		return MediaCellular
	}
	return MediaUnknown
}
