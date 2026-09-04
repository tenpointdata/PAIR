// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package peerbook

import (
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"
)

const peerB = "2222bbbb-3333-cccc-4444-dddd5555eeee"

func TestOpenOnAMissingDirectoryIsEmptyNotAnError(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "never-created")
	b, err := Open(dir)
	if err != nil {
		t.Fatalf("Open on a missing dir: %v", err)
	}
	if got := len(b.Peers()); got != 0 {
		t.Fatalf("book holds %d peers, want 0", got)
	}
	if _, err := os.Stat(dir); err == nil {
		t.Fatal("Open created the directory; nothing should exist until something is written")
	}
}

func TestFlushAndReopenRoundTrip(t *testing.T) {
	dir := t.TempDir()
	b, err := Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318, "cl": 14321})
	b.Learn(peerA, []Endpoint{
		{Kind: KindName, Host: "studio.example.net"},
		{Kind: KindMapped, Host: "198.51.100.7", Ports: map[string]int{"cl": 42021}},
	}, "manual")
	b.RecordSuccess(peerA, "studio.example.net:14321")
	if err := b.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	reopened, err := Open(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	p, ok := reopened.Peer(peerA)
	if !ok {
		t.Fatal("peer did not survive the round trip")
	}
	if p.Name != "studio" || p.Ports["cl"] != 14321 {
		t.Fatalf("identity did not survive: %+v", p)
	}
	if len(p.Endpoints) != 2 {
		t.Fatalf("endpoints = %d, want 2", len(p.Endpoints))
	}

	// The evidence is the point of persisting at all: a restart must not make a
	// node re-discover which of its peers' addresses work.
	got := reopened.Candidates(peerA, "cl")
	if len(got) == 0 || got[0] != "studio.example.net:14321" {
		t.Fatalf("Candidates after reopen = %v, want the proven address first", got)
	}
	if got[1] != "198.51.100.7:42021" {
		t.Fatalf("port override did not survive: %v", got)
	}
}

// Nothing is written until Flush, and only what changed is written. The book is
// mutated on every dial; writing per mutation would put an fsync in a routing
// decision.
func TestNothingIsWrittenBeforeFlush(t *testing.T) {
	dir := t.TempDir()
	b, _ := Open(dir)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	b.Learn(peerA, []Endpoint{{Kind: KindLocal, Host: "10.0.0.9"}}, "mdns")

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read dir: %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("directory holds %d entries before Flush, want 0", len(entries))
	}

	if err := b.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, peerA+fileExt)); err != nil {
		t.Fatalf("entry missing after Flush: %v", err)
	}
}

func TestFlushIsIdempotentAndClearsDirty(t *testing.T) {
	dir := t.TempDir()
	b, _ := Open(dir)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	if err := b.Flush(); err != nil {
		t.Fatalf("first flush: %v", err)
	}
	path := filepath.Join(dir, peerA+fileExt)
	first, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}

	if err := b.Flush(); err != nil {
		t.Fatalf("second flush: %v", err)
	}
	second, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat after second flush: %v", err)
	}
	if !first.ModTime().Equal(second.ModTime()) {
		t.Fatal("a second flush rewrote an unchanged entry")
	}
}

// A book is read at startup, right after the crash most likely to have
// interrupted a write, so a partial file must not be able to exist. This checks
// the observable half of that: no stray temporary file is left behind.
func TestFlushLeavesNoTemporaryFiles(t *testing.T) {
	dir := t.TempDir()
	b, _ := Open(dir)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	b.SetIdentity(peerB, "laptop", map[string]int{"ni": 14318})
	if err := b.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read dir: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("directory holds %d entries, want exactly the two peers: %v", len(entries), names(entries))
	}
}

func names(entries []os.DirEntry) []string {
	out := make([]string, 0, len(entries))
	for _, e := range entries {
		out = append(out, e.Name())
	}
	return out
}

// One truncated entry costs one peer a slower reconnection. Failing the open
// would cost every peer, over the file the crash most likely damaged.
func TestUnreadableEntryIsSkippedNotFatal(t *testing.T) {
	dir := t.TempDir()
	b, _ := Open(dir)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	if err := b.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, peerB+fileExt), []byte("{ truncated"), 0o600); err != nil {
		t.Fatal(err)
	}

	reopened, err := Open(dir)
	if err != nil {
		t.Fatalf("Open with a damaged entry: %v", err)
	}
	if _, ok := reopened.Peer(peerA); !ok {
		t.Fatal("the good entry was lost along with the bad one")
	}
	if _, ok := reopened.Peer(peerB); ok {
		t.Fatal("the damaged entry was loaded")
	}
}

// Files that are not entries are left strictly alone: the directory belongs to
// the application data path, and deleting or parsing a stranger's file is not
// this package's business.
func TestNonEntryFilesAreIgnored(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "README"), []byte("hello"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "subdir"), 0o700); err != nil {
		t.Fatal(err)
	}
	b, err := Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if got := len(b.Peers()); got != 0 {
		t.Fatalf("book holds %d peers, want 0", got)
	}
}

// A node UUID arrives over the network — a gossip message names the peer it
// describes — and then becomes a path. Anything that is not UUID-shaped is
// refused rather than cleaned.
func TestUnsafeUUIDsAreRefused(t *testing.T) {
	for _, uuid := range []string{
		"", "../../etc/passwd", "a/b", `a\b`, "a:b", "peer with spaces", ".",
	} {
		if safeUUID(uuid) {
			t.Errorf("safeUUID(%q) = true, want false", uuid)
		}
		if _, err := pathFor("/tmp", uuid); err == nil {
			t.Errorf("pathFor(%q) accepted an unsafe uuid", uuid)
		}
	}
	if !safeUUID(peerA) {
		t.Errorf("safeUUID(%q) = false, want true", peerA)
	}
}

// The filename is authoritative. A file whose body claims a different UUID was
// moved or hand-edited, and trusting the body would file the entry where no
// later write or delete could find it.
func TestFilenameWinsOverBodyUUID(t *testing.T) {
	dir := t.TempDir()
	body := `{"nodeUuid":"` + peerB + `","name":"impostor"}`
	if err := os.WriteFile(filepath.Join(dir, peerA+fileExt), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	b, err := Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if _, ok := b.Peer(peerA); !ok {
		t.Fatal("entry was not filed under its filename")
	}
	if _, ok := b.Peer(peerB); ok {
		t.Fatal("entry was filed under the uuid in its body")
	}
}

func TestForgetRemovesTheFile(t *testing.T) {
	dir := t.TempDir()
	b, _ := Open(dir)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	if err := b.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if err := b.Forget(peerA); err != nil {
		t.Fatalf("Forget: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, peerA+fileExt)); !os.IsNotExist(err) {
		t.Fatalf("entry still on disk: %v", err)
	}
	// Forget runs on removal paths that can fire more than once.
	if err := b.Forget(peerA); err != nil {
		t.Fatalf("second Forget: %v", err)
	}
}

func TestEntriesAreOwnerOnly(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX mode bits are not meaningful on Windows")
	}
	dir := t.TempDir()
	b, _ := Open(dir)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	if err := b.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	fi, err := os.Stat(filepath.Join(dir, peerA+fileExt))
	if err != nil {
		t.Fatal(err)
	}
	if perm := fi.Mode().Perm(); perm&^fs.FileMode(0o600) != 0 {
		t.Fatalf("entry mode is %v; a map of where a user's machines are should be owner-only", perm)
	}
}

// Learning and consuming happen in different goroutines of different relays, so
// the race detector is the assertion here.
func TestConcurrentUseIsSafe(t *testing.T) {
	dir := t.TempDir()
	b, _ := Open(dir)
	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})

	var wg sync.WaitGroup
	for i := range 6 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := range 100 {
				b.Learn(peerA, []Endpoint{{Kind: KindLocal, Host: "10.0.0." + string(rune('1'+j%9))}}, "mdns")
				b.RecordSuccess(peerA, "10.0.0.1")
				b.RecordFailure(peerA, "10.0.0.2")
				_ = b.Candidates(peerA, "ni")
				_ = b.Peers()
				if i == 0 && j%20 == 0 {
					_ = b.Flush()
				}
			}
		}(i)
	}
	wg.Wait()

	if err := b.Flush(); err != nil {
		t.Fatalf("final flush: %v", err)
	}
	if _, ok := b.Peer(peerA); !ok {
		t.Fatal("peer lost under concurrent use")
	}
}

func TestPruneMarksDirtyOnlyWhenSomethingWasDropped(t *testing.T) {
	dir := t.TempDir()
	b, _ := Open(dir)
	now := time.Date(2026, 3, 1, 9, 0, 0, 0, time.UTC)
	b.now = func() time.Time { return now }

	b.SetIdentity(peerA, "studio", map[string]int{"ni": 14318})
	b.Learn(peerA, []Endpoint{{Kind: KindLocal, Host: "10.0.0.9"}}, "mdns")
	if err := b.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	b.Prune()
	b.mu.RLock()
	dirty := len(b.dirty)
	b.mu.RUnlock()
	if dirty != 0 {
		t.Fatal("pruning nothing should not schedule a write")
	}

	now = now.Add(staleEndpoint + time.Hour)
	b.Prune()
	b.mu.RLock()
	dirty = len(b.dirty)
	b.mu.RUnlock()
	if dirty != 1 {
		t.Fatalf("dirty = %d after a real prune, want 1", dirty)
	}
}
