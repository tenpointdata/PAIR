// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"nvpair-shared/poolwire"
)

// Donation hands another machine the ability to allocate memory in a process on
// this one. Every path that cannot read a clear "yes" must answer no.
func TestDonationDefaultsOffOnEveryUnreadablePath(t *testing.T) {
	dir := t.TempDir()

	t.Run("no file", func(t *testing.T) {
		s, err := LoadDonorState(filepath.Join(dir, "absent.json"))
		if err != nil {
			t.Fatalf("LoadDonorState: %v", err)
		}
		if s.Settings().Enabled {
			t.Fatal("a node with no settings file must not be a donor")
		}
	})

	t.Run("corrupt file", func(t *testing.T) {
		path := filepath.Join(dir, "corrupt.json")
		if err := os.WriteFile(path, []byte("{ this is not json"), 0o600); err != nil {
			t.Fatal(err)
		}
		s, err := LoadDonorState(path)
		if err != nil {
			t.Fatalf("a corrupt file should not fail the process: %v", err)
		}
		if s.Settings().Enabled {
			t.Fatal("a corrupt settings file must not be read as consent")
		}
	})

	t.Run("empty file", func(t *testing.T) {
		path := filepath.Join(dir, "empty.json")
		if err := os.WriteFile(path, nil, 0o600); err != nil {
			t.Fatal(err)
		}
		s, err := LoadDonorState(path)
		if err != nil {
			t.Fatalf("LoadDonorState: %v", err)
		}
		if s.Settings().Enabled {
			t.Fatal("an empty settings file must not be read as consent")
		}
	})
}

func TestDonorSettingsSurviveARestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "pool-donor.json")
	s, err := LoadDonorState(path)
	if err != nil {
		t.Fatalf("LoadDonorState: %v", err)
	}
	want := poolwire.DonorSettings{Enabled: true, ReservedBytes: 2 << 30}
	changed, err := s.Set(want)
	if err != nil {
		t.Fatalf("Set: %v", err)
	}
	if !changed {
		t.Fatal("the first Set should report a change")
	}

	reloaded, err := LoadDonorState(path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if reloaded.Settings() != want {
		t.Fatalf("settings after restart = %+v, want %+v", reloaded.Settings(), want)
	}
}

// A no-op re-assert must not be announced as a change, or every status poll that
// echoed settings back would look like the user had just changed something.
func TestReassertingTheSameSettingsIsNotAChange(t *testing.T) {
	path := filepath.Join(t.TempDir(), "pool-donor.json")
	s, _ := LoadDonorState(path)
	want := poolwire.DonorSettings{Enabled: true, ReservedBytes: 1 << 30}

	if changed, err := s.Set(want); err != nil || !changed {
		t.Fatalf("first Set: changed=%v err=%v", changed, err)
	}
	if changed, err := s.Set(want); err != nil || changed {
		t.Fatalf("second Set: changed=%v err=%v, want no change", changed, err)
	}
}

// Bytes typed where gigabytes were meant would disable donation on every device
// while the switch still read as on — a silent failure, so it is refused loudly.
func TestImplausibleReservationIsRefused(t *testing.T) {
	path := filepath.Join(t.TempDir(), "pool-donor.json")
	s, _ := LoadDonorState(path)

	if _, err := s.Set(poolwire.DonorSettings{Enabled: true, ReservedBytes: maxReservedBytes + 1}); err == nil {
		t.Fatal("a reservation larger than any device should be refused")
	}
	if s.Settings().Enabled {
		t.Fatal("a refused Set must not have applied any part of itself")
	}
	if _, err := s.Set(poolwire.DonorSettings{Enabled: true, ReservedBytes: maxReservedBytes}); err != nil {
		t.Fatalf("a large but plausible reservation should be accepted: %v", err)
	}
}

func TestSettingsAreWrittenAtomicallyAndOwnerOnly(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "pool-donor.json")
	s, _ := LoadDonorState(path)
	if _, err := s.Set(poolwire.DonorSettings{Enabled: true}); err != nil {
		t.Fatalf("Set: %v", err)
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != "pool-donor.json" {
		names := make([]string, 0, len(entries))
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Fatalf("directory holds %v, want only the settings file", names)
	}

	if runtime.GOOS == "windows" {
		t.Skip("POSIX mode bits are not meaningful on Windows")
	}
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if perm := fi.Mode().Perm(); perm&^fs.FileMode(0o600) != 0 {
		t.Fatalf("settings mode is %v; a decision about this machine's hardware should be owner-only", perm)
	}
}

// An in-memory state (no path) is what a test and a --donor-settings-less run
// both get, and it must not try to write anywhere.
func TestPathlessStateStaysInMemory(t *testing.T) {
	s := &DonorState{}
	changed, err := s.Set(poolwire.DonorSettings{Enabled: true})
	if err != nil || !changed {
		t.Fatalf("Set on a pathless state: changed=%v err=%v", changed, err)
	}
	if !s.Settings().Enabled {
		t.Fatal("the change should still apply in memory")
	}
}

// A failed write must not leave this node advertising a policy the disk does not
// agree with. Enabling would then lend memory until the next restart and stop
// silently; disabling would keep lending after the owner was told it stopped.
func TestAFailedWriteLeavesThePolicyUnchanged(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sub", "pool-donor.json")
	s, err := LoadDonorState(path)
	if err != nil {
		t.Fatalf("LoadDonorState: %v", err)
	}

	// A regular file where the parent directory has to be makes MkdirAll fail,
	// which is the cheapest portable way to make the write fail for real rather
	// than by injecting a fake.
	if err := os.WriteFile(filepath.Join(dir, "sub"), []byte("not a directory"), 0o600); err != nil {
		t.Fatal(err)
	}

	changed, err := s.Set(poolwire.DonorSettings{Enabled: true})
	if err == nil {
		t.Fatal("Set should have failed when the settings file cannot be written")
	}
	if changed {
		t.Fatal("a failed Set must not report a change")
	}
	if s.Settings().Enabled {
		t.Fatal("a failed write must not commit the policy in memory")
	}
}
