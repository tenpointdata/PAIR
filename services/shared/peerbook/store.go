// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package peerbook

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
)

// fileExt is the suffix of one peer's entry. One file per peer, named by UUID,
// mirroring nvpair-cluster-manager's trusted/ directory — the same peers, filed
// the same way, so an operator reading one directory can read the other.
const fileExt = ".json"

// dirPerm and filePerm keep the book owner-only. Nothing in it is secret in the
// way a private key is, but it is a map of where a user's other machines are and
// when they were last seen there, and that is not something to leave
// world-readable in a shared home directory.
const (
	dirPerm  os.FileMode = 0o700
	filePerm os.FileMode = 0o600
)

// safeUUID reports whether a node UUID can be used as a filename.
//
// The book is written from data that arrived over the network — a gossip
// message names the peer it describes — so a UUID is untrusted input that
// becomes a path. Restricting it to the characters a UUID actually contains
// refuses "../../.ssh/authorized_keys" without needing to reason about path
// cleaning, which is the kind of reasoning that turns out to have been wrong.
func safeUUID(uuid string) bool {
	if uuid == "" || len(uuid) > 128 {
		return false
	}
	for _, r := range uuid {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
		case r == '-', r == '_':
		default:
			return false
		}
	}
	return true
}

// pathFor returns the file backing one peer's entry.
func pathFor(dir, uuid string) (string, error) {
	if !safeUUID(uuid) {
		return "", fmt.Errorf("peerbook: refusing unsafe node uuid %q", uuid)
	}
	return filepath.Join(dir, uuid+fileExt), nil
}

// loadAll reads every entry in dir.
//
// A missing directory is not an error: a node that has never paired has no book,
// and creating one on read would put an empty directory in the application data
// path of every install that never forms a cluster.
func loadAll(dir string) ([]*Peer, error) {
	entries, err := os.ReadDir(dir)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("peerbook: read dir: %w", err)
	}
	var out []*Peer
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), fileExt) {
			continue
		}
		uuid := strings.TrimSuffix(entry.Name(), fileExt)
		if !safeUUID(uuid) {
			slog.Warn("peerbook: skipping entry with an unusable name", "file", entry.Name())
			continue
		}
		p, err := load(dir, uuid)
		if err != nil {
			// One unreadable entry costs one peer a slower reconnection. Failing
			// the whole load would cost every peer, and the unreadable file is
			// most likely the one truncated by whatever made this a restart.
			slog.Warn("peerbook: skipping unreadable entry", "uuid", uuid, "err", err)
			continue
		}
		out = append(out, p)
	}
	return out, nil
}

// load reads one peer's entry.
func load(dir, uuid string) (*Peer, error) {
	path, err := pathFor(dir, uuid)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var p Peer
	if err := json.Unmarshal(data, &p); err != nil {
		return nil, fmt.Errorf("parse %s: %w", filepath.Base(path), err)
	}
	// The filename is authoritative over the body. They agree in every file this
	// package writes; when they disagree, the file was moved or hand-edited, and
	// keying the in-memory book by the body would file the entry where no
	// subsequent write or delete would find it.
	p.NodeUUID = uuid
	return &p, nil
}

// save writes one peer's entry atomically: a temporary file in the same
// directory, then a rename.
//
// Same directory because rename is only atomic within a filesystem, and a
// temporary directory can be on another one. Atomic at all because a book is
// read at startup — exactly after the crash most likely to have interrupted a
// write — and a half-written entry would be a peer this node cannot find until
// somebody deletes a file by hand.
func save(dir string, p Peer) error {
	path, err := pathFor(dir, p.NodeUUID)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, dirPerm); err != nil {
		return fmt.Errorf("peerbook: create dir: %w", err)
	}
	data, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return fmt.Errorf("peerbook: encode %s: %w", p.NodeUUID, err)
	}
	tmp, err := os.CreateTemp(dir, "."+p.NodeUUID+".*")
	if err != nil {
		return fmt.Errorf("peerbook: temp file: %w", err)
	}
	tmpName := tmp.Name()
	defer func() { _ = os.Remove(tmpName) }()

	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("peerbook: write %s: %w", p.NodeUUID, err)
	}
	if err := tmp.Chmod(filePerm); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("peerbook: chmod %s: %w", p.NodeUUID, err)
	}
	// Sync before rename: a rename that lands before the data does leaves a
	// valid directory entry pointing at a file whose contents were never
	// written, which is precisely the state this whole dance exists to avoid.
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("peerbook: sync %s: %w", p.NodeUUID, err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("peerbook: close %s: %w", p.NodeUUID, err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("peerbook: rename %s: %w", p.NodeUUID, err)
	}
	return nil
}

// remove deletes one peer's entry. A missing file is success: Forget is called
// on removal paths that can run more than once.
func remove(dir, uuid string) error {
	path, err := pathFor(dir, uuid)
	if err != nil {
		return err
	}
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("peerbook: remove %s: %w", uuid, err)
	}
	return nil
}
