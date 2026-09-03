// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package linkq

import (
	"os"
	"path/filepath"
	"testing"
)

// fakeIface writes one interface into a fixture sysfs tree: its ARPHRD type, and
// whichever of the attribute directories the case needs.
func fakeIface(t *testing.T, root, name, arphrd string, attrs ...string) {
	t.Helper()
	base := filepath.Join(root, name)
	if err := os.MkdirAll(base, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", base, err)
	}
	if arphrd != "" {
		if err := os.WriteFile(filepath.Join(base, "type"), []byte(arphrd+"\n"), 0o644); err != nil {
			t.Fatalf("write type: %v", err)
		}
	}
	for _, a := range attrs {
		if err := os.MkdirAll(filepath.Join(base, a), 0o755); err != nil {
			t.Fatalf("mkdir attr %s: %v", a, err)
		}
	}
}

// The fixture tree stands in for /sys/class/net, so the detector's real logic
// runs against interfaces this machine does not have — a Wi-Fi adapter, a
// modem — which is the whole reason the path is a variable.
func TestInterfaceMediaFromSysfs(t *testing.T) {
	root := t.TempDir()
	orig := sysClassNet
	sysClassNet = root
	t.Cleanup(func() { sysClassNet = orig })

	fakeIface(t, root, "lo", "772")
	fakeIface(t, root, "eth0", "1", "device")
	fakeIface(t, root, "wlan0", "1", "device", "phy80211")
	fakeIface(t, root, "wlan1", "1", "device", "wireless") // pre-cfg80211 attribute
	fakeIface(t, root, "docker0", "1")                     // no backing device
	fakeIface(t, root, "wg0", "65534", "device")           // named virtual with a device
	fakeIface(t, root, "ppp0", "512", "device")
	fakeIface(t, root, "rmnet_data0", "519", "device")
	fakeIface(t, root, "wwan0", "1", "device")
	fakeIface(t, root, "enp3s0", "")

	cases := map[string]Media{
		"lo":          MediaLoopback,
		"eth0":        MediaWired,
		"wlan0":       MediaWireless,
		"wlan1":       MediaWireless,
		"docker0":     MediaVirtual,
		"wg0":         MediaVirtual,
		"ppp0":        MediaCellular,
		"rmnet_data0": MediaCellular,
		"wwan0":       MediaCellular,
		// No type attribute and no backing device: everything sysfs can say
		// points at a software interface, so it reads virtual rather than being
		// offered to the scheduler as a cable.
		"enp3s0": MediaVirtual,
		// An interface that is simply not there.
		"absent0": MediaUnknown,
	}

	for name, want := range cases {
		if got := interfaceMedia(name); got != want {
			t.Errorf("interfaceMedia(%q) = %v, want %v", name, got, want)
		}
	}
}

// An interface that vanishes between enumeration and classification is an
// ordinary event on a machine that just changed networks, and it must read as
// unknown rather than panic or block.
func TestInterfaceMediaOnMissingTreeIsUnknown(t *testing.T) {
	orig := sysClassNet
	sysClassNet = filepath.Join(t.TempDir(), "does-not-exist")
	t.Cleanup(func() { sysClassNet = orig })

	if got := interfaceMedia("eth0"); got != MediaUnknown {
		t.Fatalf("interfaceMedia on a missing tree = %v, want %v", got, MediaUnknown)
	}
}

func TestReadSysIntHandlesGarbage(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "good")
	bad := filepath.Join(dir, "bad")
	if err := os.WriteFile(good, []byte(" 772 \n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(bad, []byte("not-a-number"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := readSysInt(good); got != 772 {
		t.Fatalf("readSysInt(good) = %d, want 772", got)
	}
	if got := readSysInt(bad); got != -1 {
		t.Fatalf("readSysInt(bad) = %d, want -1", got)
	}
	if got := readSysInt(filepath.Join(dir, "missing")); got != -1 {
		t.Fatalf("readSysInt(missing) = %d, want -1", got)
	}
}
