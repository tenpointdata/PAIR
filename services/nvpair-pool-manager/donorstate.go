// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"nvpair-shared/poolwire"
)

// donorFilePerm keeps the settings file owner-only. It records a decision about
// this machine's hardware, and on a shared host that is nobody else's to read or
// to change.
const donorFilePerm os.FileMode = 0o600

// maxReservedBytes bounds the per-device reservation a caller may set. A
// terabyte is far beyond any device that exists, so anything larger is a unit
// error — bytes typed where gigabytes were meant — and accepting it would
// silently disable donation on every device while the switch still read as on.
const maxReservedBytes = 1 << 40

// DonorState is this node's donation policy, held in memory and persisted on
// change.
//
// It is deliberately small, and deliberately its own file rather than a key in
// nvpair-node-settings. Donation is the one setting in PAIR that hands another
// machine the ability to allocate memory in a process on this one, so it is
// worth being able to answer "what did this node agree to, and when" by reading
// a single file — and worth having that answer survive a settings store that is
// reset, migrated, or shared.
type DonorState struct {
	mu       sync.RWMutex
	path     string
	settings poolwire.DonorSettings
}

// LoadDonorState reads path, or returns the default policy when it does not
// exist.
//
// The default is donation OFF. That is the whole reason this type has a
// constructor rather than a zero value: an unreadable or absent file must fail
// closed, and a bug that made it fail open would enrol every node in the cluster
// as a donor without anyone choosing to.
func LoadDonorState(path string) (*DonorState, error) {
	s := &DonorState{path: path}
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return s, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read donor settings: %w", err)
	}
	var settings poolwire.DonorSettings
	if err := json.Unmarshal(data, &settings); err != nil {
		// A corrupt file reads as the default rather than failing the process.
		// The safe reading and the conservative one are the same here — donation
		// off — so there is nothing to gain by refusing to start.
		return s, nil
	}
	s.settings = settings
	return s, nil
}

// Settings returns the current policy.
func (s *DonorState) Settings() poolwire.DonorSettings {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.settings
}

// Set replaces the policy and persists it. It reports whether anything actually
// changed, so a caller does not announce an update that was a no-op re-assert.
func (s *DonorState) Set(next poolwire.DonorSettings) (changed bool, err error) {
	if next.ReservedBytes > maxReservedBytes {
		return false, fmt.Errorf("reserved bytes %d is implausible; expected a per-device figure in bytes", next.ReservedBytes)
	}
	s.mu.RLock()
	unchanged := s.settings == next
	path := s.path
	s.mu.RUnlock()
	if unchanged {
		return false, nil
	}

	// Persist BEFORE committing in memory, so a failed write never leaves this
	// node advertising a policy the disk does not agree with. For this setting
	// that gap is not cosmetic: a failed write while enabling means the node
	// lends its memory until the next restart and then silently stops, and a
	// failed write while DISABLING means it keeps lending after the owner was
	// told it had stopped. Both are worse than reporting the failure.
	//
	// Safe as a read-modify-write because the caller handles one JSON-RPC frame
	// at a time off the read loop; no second setter can interleave here.
	if path != "" {
		if err := writeDonorSettings(path, next); err != nil {
			return false, err
		}
	}
	s.mu.Lock()
	s.settings = next
	s.mu.Unlock()
	return true, nil
}

// writeDonorSettings persists the policy atomically: a temporary file in the
// same directory, then a rename.
//
// Same directory because rename is only atomic within a filesystem. Atomic at
// all because this file is read at startup to decide whether this machine lends
// its memory, and a half-written file that happened to parse is a decision
// nobody made.
func writeDonorSettings(path string, settings poolwire.DonorSettings) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("create donor settings dir: %w", err)
	}
	data, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return fmt.Errorf("encode donor settings: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".pool-donor.*")
	if err != nil {
		return fmt.Errorf("create temp donor settings: %w", err)
	}
	tmpName := tmp.Name()
	defer func() { _ = os.Remove(tmpName) }()

	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("write donor settings: %w", err)
	}
	if err := tmp.Chmod(donorFilePerm); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("chmod donor settings: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("sync donor settings: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close donor settings: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("rename donor settings: %w", err)
	}
	return nil
}
