// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package linkq

import (
	"errors"
	"net"
	"unsafe"

	"golang.org/x/sys/windows"
)

// IANA ifType values for mobile broadband. x/sys/windows defines the types this
// package needs except these two, so they are spelled out here rather than
// having the detector silently miss a tethered link.
const (
	ifTypeWWANPP  = 243 // GSM
	ifTypeWWANPP2 = 244 // CDMA
)

// adapterBufferHint is the first buffer size tried for GetAdaptersAddresses.
// Microsoft's own guidance is 15 kB, which covers an ordinary host in one call;
// a machine with many virtual adapters grows past it and is handled by the retry.
const adapterBufferHint = 15000

// adapterBufferAttempts bounds the grow-and-retry loop. The size the API returns
// on overflow is exact, so two attempts always suffice in practice; the third
// exists only because the adapter list can change between the two calls.
const adapterBufferAttempts = 3

// interfaceMedia classifies a Windows interface from the ifType the IP Helper
// API reports.
//
// Windows is the platform where this is least ambiguous: ifType is a first-class
// property maintained by the OS, so unlike Linux there is no need to infer a
// wireless link from the presence of a driver attribute. The one judgement call
// left is that Hyper-V and similar virtual switches report themselves as
// ordinary Ethernet, so an Ethernet answer is checked against the software
// interface names before being believed.
func interfaceMedia(name string) Media {
	iface, err := net.InterfaceByName(name)
	if err != nil {
		return MediaUnknown
	}
	if iface.Flags&net.FlagLoopback != 0 {
		return MediaLoopback
	}
	head, err := adapterAddresses()
	if err != nil {
		return MediaUnknown
	}
	for p := head; p != nil; p = p.Next {
		if int(p.IfIndex) != iface.Index && int(p.Ipv6IfIndex) != iface.Index {
			continue
		}
		return mediaFromIfType(p.IfType, name)
	}
	return MediaUnknown
}

// mediaFromIfType maps an IANA ifType to a Media.
func mediaFromIfType(ifType uint32, name string) Media {
	switch ifType {
	case windows.IF_TYPE_SOFTWARE_LOOPBACK:
		return MediaLoopback
	case windows.IF_TYPE_IEEE80211:
		return MediaWireless
	case ifTypeWWANPP, ifTypeWWANPP2, windows.IF_TYPE_PPP:
		return MediaCellular
	case windows.IF_TYPE_TUNNEL:
		return MediaVirtual
	case windows.IF_TYPE_ETHERNET_CSMACD, windows.IF_TYPE_IEEE1394:
		if looksVirtual(name) {
			return MediaVirtual
		}
		return MediaWired
	default:
		return MediaUnknown
	}
}

// adapterAddresses returns the head of the OS adapter list.
//
// The unicast, anycast, multicast, and DNS sub-lists are skipped: this detector
// needs one integer per adapter, and asking for the address lists as well
// multiplies the buffer on a host with many interfaces for data nothing here
// reads.
func adapterAddresses() (*windows.IpAdapterAddresses, error) {
	const flags = windows.GAA_FLAG_SKIP_UNICAST |
		windows.GAA_FLAG_SKIP_ANYCAST |
		windows.GAA_FLAG_SKIP_MULTICAST |
		windows.GAA_FLAG_SKIP_DNS_SERVER

	size := uint32(adapterBufferHint)
	for range adapterBufferAttempts {
		buf := make([]byte, size)
		head := (*windows.IpAdapterAddresses)(unsafe.Pointer(&buf[0]))
		err := windows.GetAdaptersAddresses(windows.AF_UNSPEC, flags, 0, head, &size)
		if err == nil {
			return head, nil
		}
		if !errors.Is(err, windows.ERROR_BUFFER_OVERFLOW) {
			return nil, err
		}
		// size now holds the exact requirement; loop and allocate it.
	}
	return nil, errors.New("linkq: adapter list kept growing")
}
