// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package peerbook

import (
	"maps"
	"sort"
	"strings"
	"sync"
	"time"
)

// Book is the address book: every paired peer's recorded endpoints, the evidence
// for each, and the order to try them in.
//
// It is safe for concurrent use, which is not optional — the services that learn
// endpoints (discovery, roster reconcile, manual entry) are not the services that
// consume them (every peer dialer in the tree), and they run in different
// goroutines of different processes' relays.
//
// Writes to disk are explicit. Endpoint evidence changes on every dial in a busy
// cluster, and writing a file per dial would turn a routing decision into an
// fsync; instead a mutation marks its peer dirty and Flush persists what changed.
// The cost of that choice is bounded and known: a crash loses at most the
// evidence gathered since the last flush, which is re-earned by the next
// successful connection.
type Book struct {
	mu    sync.RWMutex
	peers map[string]*Peer
	dirty map[string]struct{}

	// dir is where entries persist. Empty means memory-only, which is what a
	// test and an unclustered node both want.
	dir string

	// now reads the clock. Overridden in tests.
	now func() time.Time
}

// New returns an empty in-memory book. Nothing it records survives the process.
func New() *Book {
	return &Book{
		peers: make(map[string]*Peer),
		dirty: make(map[string]struct{}),
		now:   time.Now,
	}
}

// Open returns a book backed by dir, loading whatever is already there.
//
// A file that cannot be read or parsed is skipped rather than failing the open.
// The book is a cache of locations, not a record of membership: losing one
// peer's entry costs a slower reconnection to that peer, while refusing to open
// costs every peer's entry over one bad file — and the bad file is most likely
// the one truncated by the crash that made this restart necessary.
func Open(dir string) (*Book, error) {
	b := New()
	b.dir = dir
	if dir == "" {
		return b, nil
	}
	peers, err := loadAll(dir)
	if err != nil {
		return nil, err
	}
	for _, p := range peers {
		b.peers[p.NodeUUID] = p
	}
	return b, nil
}

// peer returns a peer entry, creating it on first mention. Callers hold the
// write lock. A blank UUID is refused: the book is keyed by identity, and an
// entry under "" would collect every unidentified sighting in the cluster into
// one peer that is no peer at all.
func (b *Book) peer(nodeUUID string) *Peer {
	if nodeUUID == "" {
		return nil
	}
	p, ok := b.peers[nodeUUID]
	if !ok {
		p = &Peer{NodeUUID: nodeUUID}
		b.peers[nodeUUID] = p
	}
	return p
}

// touch marks a peer changed and due for persistence. Callers hold the write
// lock.
func (b *Book) touch(p *Peer) {
	p.UpdatedAt = b.now()
	b.dirty[p.NodeUUID] = struct{}{}
}

// SetIdentity records a peer's friendly name and the service ports it
// advertises. Ports are replaced wholesale rather than merged: the advertisement
// is the peer's current, complete statement about what it serves, and merging
// would keep a port alive after the service behind it was removed.
func (b *Book) SetIdentity(nodeUUID, name string, ports map[string]int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	p := b.peer(nodeUUID)
	if p == nil {
		return
	}
	if name != "" {
		p.Name = name
	}
	if ports != nil {
		p.Ports = maps.Clone(ports)
	}
	b.touch(p)
}

// Learn merges advertised endpoints for a peer, attributing them to source.
//
// Merging rather than replacing is the whole point of a book. An advertisement
// describes where the peer believes it is right now, from one vantage point; it
// is not a statement that everywhere it used to be is gone. A laptop that
// announces one hotel address should not lose the office address it will be back
// at tomorrow, and a gossip relay that knows only one of a peer's endpoints
// should not erase the rest.
//
// What the merge does update is recency: an endpoint mentioned again is
// re-advertised now, which is what keeps it ahead of one nobody has mentioned in
// a month, and what stops pruning from taking it.
func (b *Book) Learn(nodeUUID string, endpoints []Endpoint, source string) {
	if len(endpoints) == 0 {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	p := b.peer(nodeUUID)
	if p == nil {
		return
	}
	now := b.now()
	existing := make(map[string]int, len(p.Endpoints))
	for i, e := range p.Endpoints {
		existing[e.key()] = i
	}
	for _, in := range endpoints {
		if in.Host == "" {
			continue
		}
		if in.Kind == KindRelay && in.Via == "" {
			// A relay endpoint that does not say which member relays it cannot
			// be used and cannot be re-learned correctly later.
			continue
		}
		in.Source = source
		in.LastAdvertised = now
		if i, ok := existing[in.key()]; ok {
			cur := &p.Endpoints[i]
			cur.Source = source
			cur.LastAdvertised = now
			if len(in.Ports) > 0 {
				cur.Ports = maps.Clone(in.Ports)
			}
			continue
		}
		in.FirstSeen = now
		in.Ports = maps.Clone(in.Ports)
		p.Endpoints = append(p.Endpoints, in)
		existing[in.key()] = len(p.Endpoints) - 1
	}
	b.capEndpoints(p, now)
	b.touch(p)
}

// capEndpoints enforces MaxEndpointsPerPeer by dropping the worst-ranked
// entries. Callers hold the write lock.
//
// Eviction reuses the candidate ordering rather than dropping the oldest,
// because age is not the thing that makes an endpoint worthless. The oldest
// entry may be the peer's static home address, which is the one that will work
// tomorrow; the entry worth losing is the one that has failed repeatedly and
// which nothing has advertised since.
func (b *Book) capEndpoints(p *Peer, now time.Time) {
	if len(p.Endpoints) <= MaxEndpointsPerPeer {
		return
	}
	ranked := sortEndpoints(p.Endpoints, now)
	p.Endpoints = ranked[:MaxEndpointsPerPeer]
}

// RecordSuccess notes that a connection to addr completed. Pass back what
// Candidates handed out — a host:port pair, or a bare host; both are accepted so
// a caller never has to take an address apart to report on it.
//
// A success clears the consecutive-failure count outright. The count exists to
// demote an address that has stopped working, and one completed handshake proves
// it is working now; keeping a penalty from a network the peer has since left
// would demote its best address for the rest of the day.
func (b *Book) RecordSuccess(nodeUUID, addr string) {
	b.record(nodeUUID, addr, true)
}

// RecordFailure notes that a connection to addr did not complete. Report only
// attempts that actually tried — a dial that was refused, timed out, or reset —
// never a candidate that was skipped because an earlier one answered first.
func (b *Book) RecordFailure(nodeUUID, addr string) {
	b.record(nodeUUID, addr, false)
}

func (b *Book) record(nodeUUID, addr string, ok bool) {
	host := hostOf(addr)
	if host == "" {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	p, exists := b.peers[nodeUUID]
	if !exists {
		return
	}
	now := b.now()
	changed := false
	for i := range p.Endpoints {
		if !strings.EqualFold(p.Endpoints[i].Host, host) {
			continue
		}
		e := &p.Endpoints[i]
		if ok {
			e.LastSuccess = now
			e.Successes++
			e.Failures = 0
		} else {
			e.LastFailure = now
			e.Failures++
		}
		changed = true
	}
	if changed {
		b.touch(p)
	}
}

// Candidates returns the addresses to try for one of a peer's services, best
// first, capped at MaxCandidates.
//
// Relays are excluded. Reaching a peer through a third machine spends that
// machine's bandwidth and doubles the path, so it is a fallback taken
// deliberately after the direct list is exhausted — see Relays — rather than one
// mixed into the same sweep. shared/reach probes candidates concurrently, and
// concurrently probing a relay is asking a peer to carry traffic that a direct
// endpoint was about to make unnecessary.
func (b *Book) Candidates(nodeUUID, service string) []string {
	b.mu.RLock()
	defer b.mu.RUnlock()
	p, ok := b.peers[nodeUUID]
	if !ok {
		return nil
	}
	now := b.now()
	out := make([]string, 0, MaxCandidates)
	seen := make(map[string]struct{}, MaxCandidates)
	for _, e := range sortEndpoints(p.Endpoints, now) {
		if e.Kind == KindRelay {
			continue
		}
		addr, ok := e.address(service, p.Ports)
		if !ok {
			continue
		}
		if _, dup := seen[addr]; dup {
			continue
		}
		seen[addr] = struct{}{}
		out = append(out, addr)
		if len(out) == MaxCandidates {
			break
		}
	}
	return out
}

// Relays returns the members recorded as able to reach this peer, best first.
// The caller decides whether to use one, because the cost is borne by a third
// machine rather than by either end.
func (b *Book) Relays(nodeUUID string) []Endpoint {
	b.mu.RLock()
	defer b.mu.RUnlock()
	p, ok := b.peers[nodeUUID]
	if !ok {
		return nil
	}
	var out []Endpoint
	for _, e := range sortEndpoints(p.Endpoints, b.now()) {
		if e.Kind == KindRelay {
			e.Ports = maps.Clone(e.Ports)
			out = append(out, e)
		}
	}
	return out
}

// Peer returns a copy of one entry. ok is false for a peer the book has never
// recorded, which is a different answer from a peer with no working endpoints.
func (b *Book) Peer(nodeUUID string) (Peer, bool) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	p, ok := b.peers[nodeUUID]
	if !ok {
		return Peer{}, false
	}
	return p.clone(), true
}

// Peers returns a copy of every entry, ordered by UUID so a rendering of the
// book is stable between reads.
func (b *Book) Peers() []Peer {
	b.mu.RLock()
	defer b.mu.RUnlock()
	out := make([]Peer, 0, len(b.peers))
	for _, p := range b.peers {
		out = append(out, p.clone())
	}
	sort.Slice(out, func(i, j int) bool { return out[i].NodeUUID < out[j].NodeUUID })
	return out
}

// Forget drops a peer entirely, on disk as well as in memory. Call it when a node
// is removed from the cluster or leaves — never when it merely goes quiet, which
// is the case the book exists to survive.
func (b *Book) Forget(nodeUUID string) error {
	b.mu.Lock()
	delete(b.peers, nodeUUID)
	delete(b.dirty, nodeUUID)
	dir := b.dir
	b.mu.Unlock()
	if dir == "" {
		return nil
	}
	return remove(dir, nodeUUID)
}

// Prune drops endpoints that nothing has advertised and nothing has reached for
// staleEndpoint, leaving peers themselves alone.
//
// It is separate from Forget because the two answer different questions. A peer
// with no usable endpoints is still a member and still worth an entry — the next
// gossip round may hand it a working one. An address nobody has mentioned in a
// month is a machine that has moved, and keeping it costs a dial timeout every
// time that peer is looked up.
func (b *Book) Prune() {
	b.mu.Lock()
	defer b.mu.Unlock()
	now := b.now()
	for _, p := range b.peers {
		kept := p.Endpoints[:0]
		for _, e := range p.Endpoints {
			last := e.lastEvidence()
			if !last.IsZero() && now.Sub(last) > staleEndpoint {
				continue
			}
			kept = append(kept, e)
		}
		if len(kept) != len(p.Endpoints) {
			p.Endpoints = kept
			b.touch(p)
		}
	}
}

// Flush writes every entry changed since the last call. A peer whose write fails
// stays dirty so the next flush retries it, and the first error is returned once
// every peer has been attempted — a single unwritable file must not stop the
// rest of the book from persisting.
func (b *Book) Flush() error {
	b.mu.Lock()
	dir := b.dir
	if dir == "" || len(b.dirty) == 0 {
		clear(b.dirty)
		b.mu.Unlock()
		return nil
	}
	pending := make([]Peer, 0, len(b.dirty))
	for uuid := range b.dirty {
		if p, ok := b.peers[uuid]; ok {
			pending = append(pending, p.clone())
		}
		delete(b.dirty, uuid)
	}
	b.mu.Unlock()

	var firstErr error
	for _, p := range pending {
		if err := save(dir, p); err != nil {
			if firstErr == nil {
				firstErr = err
			}
			b.mu.Lock()
			b.dirty[p.NodeUUID] = struct{}{}
			b.mu.Unlock()
		}
	}
	return firstErr
}

// hostOf extracts the host from an address that may or may not carry a port, and
// may or may not be an IPv6 literal in brackets.
//
// It is written out rather than delegating to net.SplitHostPort because that
// function is an error for a bare host, and a bare host is a perfectly ordinary
// thing for a caller to report a success on.
func hostOf(addr string) string {
	s := strings.TrimSpace(addr)
	if s == "" {
		return ""
	}
	if strings.HasPrefix(s, "[") {
		if end := strings.Index(s, "]"); end > 0 {
			return s[1:end]
		}
		return ""
	}
	// A bare IPv6 literal has several colons; only a single trailing colon
	// group is a port.
	if strings.Count(s, ":") == 1 {
		if i := strings.LastIndex(s, ":"); i >= 0 {
			return s[:i]
		}
	}
	return s
}
