// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/tls"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"path/filepath"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	"nvpair-shared/poolwire"
)

// donorRig is a paired head and donor: the donor serves capacity and leases and
// runs an echo backend on loopback, and the head holds the mesh that can reach
// it.
type donorRig struct {
	headMesh  *clustertrust.Mesh
	donorAddr string
	leases    *LeaseStore
	target    *echoTarget
}

func newDonorRig(t *testing.T) *donorRig {
	t.Helper()
	root := t.TempDir()
	headDir := filepath.Join(root, "head")
	donorDir := filepath.Join(root, "donor")
	clustertrusttest.Pair(t, "cluster-1", headDir, donorDir, "uuid-head", "uuid-donor")

	target := newEchoTarget(t)
	leases := NewLeaseStore(target)
	donorMesh := clustertrust.Open(donorDir)
	collector := NewCollector(fakeNodeInfo(t, twoGPUInventory), donorEnabled(t, 0), leases, "donor-fallback")
	addr := startPoolServer(t, donorMesh, collector, leases)

	return &donorRig{
		headMesh:  clustertrust.Open(headDir),
		donorAddr: addr,
		leases:    leases,
		target:    target,
	}
}

// headClient returns an mTLS client the head uses for lease calls.
func (r *donorRig) headClient(t *testing.T) *http.Client {
	t.Helper()
	cfg, ok := r.headMesh.ClientTLSConfig("uuid-donor")
	if !ok {
		t.Fatal("head holds no pin for the donor")
	}
	return &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{TLSClientConfig: cfg}}
}

// requestLease asks the donor for a lease over the real HTTP surface.
func (r *donorRig) requestLease(t *testing.T, poolID string, devices []int, size uint64) poolwire.LeaseGrant {
	t.Helper()
	body, err := json.Marshal(poolwire.LeaseRequest{
		PoolID: poolID, ModelName: "test-model", DeviceIndexes: devices, Bytes: size,
	})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := r.headClient(t).Post("https://"+r.donorAddr+poolwire.LeasesPath, "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("lease request: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		t.Fatalf("lease request returned %s: %s", resp.Status, msg)
	}
	var grant poolwire.LeaseGrant
	if err := json.NewDecoder(resp.Body).Decode(&grant); err != nil {
		t.Fatalf("decode grant: %v", err)
	}
	return grant
}

// The whole point of the feature: llama.cpp connects to a loopback address on
// its own machine, and the bytes come out at a loopback address on another one,
// having crossed the network only inside cluster mTLS.
func TestTunnelCarriesBytesEndToEnd(t *testing.T) {
	rig := newDonorRig(t)
	grant := rig.requestLease(t, "pool-1", []int{0}, 4*gb)

	link := NewDonorLink(rig.headMesh, "uuid-donor", rig.donorAddr, grant.LeaseID)
	if err := link.Start(context.Background()); err != nil {
		t.Fatalf("start donor link: %v", err)
	}
	defer func() { _ = link.Close() }()

	// The address handed to llama.cpp must be loopback, or the plaintext ggml
	// protocol has somewhere to leak to.
	host, _, err := net.SplitHostPort(link.Addr())
	if err != nil {
		t.Fatalf("link address %q: %v", link.Addr(), err)
	}
	if ip := net.ParseIP(host); ip == nil || !ip.IsLoopback() {
		t.Fatalf("donor link bound %s; it must be loopback", link.Addr())
	}

	conn, err := net.DialTimeout("tcp", link.Addr(), 5*time.Second)
	if err != nil {
		t.Fatalf("dial donor link: %v", err)
	}
	defer func() { _ = conn.Close() }()

	// Big enough to span many reads and to outrun any single buffer on the path,
	// because a splice that silently drops or reorders a buffered chunk is
	// exactly the bug that would corrupt a tensor transfer.
	payload := make([]byte, 3<<20)
	if _, err := rand.Read(payload); err != nil {
		t.Fatal(err)
	}

	errCh := make(chan error, 1)
	go func() {
		_, err := conn.Write(payload)
		errCh <- err
	}()

	got := make([]byte, len(payload))
	if err := conn.SetReadDeadline(time.Now().Add(30 * time.Second)); err != nil {
		t.Fatal(err)
	}
	if _, err := io.ReadFull(conn, got); err != nil {
		t.Fatalf("read echo: %v", err)
	}
	if err := <-errCh; err != nil {
		t.Fatalf("write: %v", err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatal("bytes came back altered through the tunnel")
	}
}

// tunnelAttempt performs the raw upgrade with a caller-supplied TLS config and
// lease id, returning the HTTP status the donor answered with.
func tunnelAttempt(t *testing.T, cfg *tls.Config, addr, leaseID string, upgrade bool) int {
	t.Helper()
	conn, err := tls.Dial("tcp", addr, cfg)
	if err != nil {
		return 0
	}
	defer func() { _ = conn.Close() }()

	req, err := http.NewRequest(http.MethodPost, "https://"+addr+poolwire.TunnelPathPrefix+leaseID, nil)
	if err != nil {
		t.Fatal(err)
	}
	if upgrade {
		req.Header.Set("Connection", "Upgrade")
		req.Header.Set("Upgrade", poolwire.TunnelProtocol)
	}
	if err := conn.SetDeadline(time.Now().Add(10 * time.Second)); err != nil {
		t.Fatal(err)
	}
	if err := req.Write(conn); err != nil {
		t.Fatalf("write upgrade: %v", err)
	}
	resp, err := http.ReadResponse(newBufReader(conn), req)
	if err != nil {
		t.Fatalf("read response: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	return resp.StatusCode
}

// A pinned cluster member is authorized to ASK for a lease. That is not the same
// as being authorized to use one somebody else holds — without this check any
// member that learned a lease id would be spliced into another member's tensor
// backend.
func TestTunnelRefusesAMemberWhoDoesNotHoldTheLease(t *testing.T) {
	root := t.TempDir()
	headDir := filepath.Join(root, "head")
	donorDir := filepath.Join(root, "donor")
	otherDir := filepath.Join(root, "other")

	headNode, donorNode := clustertrusttest.Pair(t, "cluster-1", headDir, donorDir, "uuid-head", "uuid-donor")

	// A third member of the same cluster: pinned by the donor, so it passes the
	// cluster gate, but it holds no lease.
	otherNode := clustertrusttest.MintNode(t, "uuid-other")
	clustertrusttest.WriteNode(t, otherDir, "cluster-1", otherNode)
	clustertrusttest.PinNode(t, otherDir, donorNode)
	clustertrusttest.PinNode(t, donorDir, otherNode)
	_ = headNode

	target := newEchoTarget(t)
	leases := NewLeaseStore(target)
	donorMesh := clustertrust.Open(donorDir)
	collector := NewCollector(fakeNodeInfo(t, twoGPUInventory), donorEnabled(t, 0), leases, "donor")
	addr := startPoolServer(t, donorMesh, collector, leases)

	rig := &donorRig{headMesh: clustertrust.Open(headDir), donorAddr: addr, leases: leases, target: target}
	grant := rig.requestLease(t, "pool-1", []int{0}, gb)

	otherMesh := clustertrust.Open(otherDir)
	cfg, ok := otherMesh.ClientTLSConfig("uuid-donor")
	if !ok {
		t.Fatal("the third member should be able to build a client")
	}
	if code := tunnelAttempt(t, cfg, addr, grant.LeaseID, true); code != http.StatusNotFound {
		t.Fatalf("a member who holds no lease got %d, want 404", code)
	}
}

// The cluster gate runs before anything else on this port, so a stranger never
// reaches the lease lookup at all.
func TestTunnelRefusesAnUnpinnedCaller(t *testing.T) {
	root := t.TempDir()
	headDir := filepath.Join(root, "head")
	donorDir := filepath.Join(root, "donor")
	strangerDir := filepath.Join(root, "stranger")
	_, donorNode := clustertrusttest.Pair(t, "cluster-1", headDir, donorDir, "uuid-head", "uuid-donor")

	strangerNode := clustertrusttest.MintNode(t, "uuid-stranger")
	clustertrusttest.WriteNode(t, strangerDir, "cluster-2", strangerNode)
	clustertrusttest.PinNode(t, strangerDir, donorNode)

	target := newEchoTarget(t)
	leases := NewLeaseStore(target)
	collector := NewCollector(fakeNodeInfo(t, twoGPUInventory), donorEnabled(t, 0), leases, "donor")
	addr := startPoolServer(t, clustertrust.Open(donorDir), collector, leases)

	rig := &donorRig{headMesh: clustertrust.Open(headDir), donorAddr: addr, leases: leases, target: target}
	grant := rig.requestLease(t, "pool-1", []int{0}, gb)

	cfg, ok := clustertrust.Open(strangerDir).ClientTLSConfig("uuid-donor")
	if !ok {
		t.Fatal("the stranger should be able to build a client")
	}
	if code := tunnelAttempt(t, cfg, addr, grant.LeaseID, true); code != http.StatusForbidden {
		t.Fatalf("an unpinned caller got %d, want 403", code)
	}
}

// A request without the protocol token is not a pool tunnel. Splicing it into a
// tensor backend on the strength of the path alone would be the wrong kind of
// forgiving.
func TestTunnelRequiresTheProtocolToken(t *testing.T) {
	rig := newDonorRig(t)
	grant := rig.requestLease(t, "pool-1", []int{0}, gb)

	cfg, ok := rig.headMesh.ClientTLSConfig("uuid-donor")
	if !ok {
		t.Fatal("no client config")
	}
	if code := tunnelAttempt(t, cfg, rig.donorAddr, grant.LeaseID, false); code != http.StatusBadRequest {
		t.Fatalf("a request without an upgrade got %d, want 400", code)
	}
}

// A lease id is a lookup key for a capability, and a lenient path parse is how
// one request becomes a lease somebody else holds.
func TestTunnelRefusesAMalformedLeasePath(t *testing.T) {
	rig := newDonorRig(t)
	cfg, ok := rig.headMesh.ClientTLSConfig("uuid-donor")
	if !ok {
		t.Fatal("no client config")
	}
	for _, id := range []string{"", "abc/def"} {
		if code := tunnelAttempt(t, cfg, rig.donorAddr, id, true); code == http.StatusSwitchingProtocols {
			t.Fatalf("lease path %q was accepted", id)
		}
	}
}

// Releasing a lease must sever live streams, or a torn-down pool leaves a tensor
// backend attached to a head that has moved on.
func TestReleasingALeaseClosesItsLiveTunnels(t *testing.T) {
	rig := newDonorRig(t)
	grant := rig.requestLease(t, "pool-1", []int{0}, gb)

	link := NewDonorLink(rig.headMesh, "uuid-donor", rig.donorAddr, grant.LeaseID)
	if err := link.Start(context.Background()); err != nil {
		t.Fatalf("start: %v", err)
	}
	defer func() { _ = link.Close() }()

	conn, err := net.DialTimeout("tcp", link.Addr(), 5*time.Second)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer func() { _ = conn.Close() }()

	// Round-trip one byte so the stream is provably established before it is torn
	// down; otherwise the test could pass on a tunnel that never opened.
	if _, err := conn.Write([]byte{0x42}); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := conn.SetReadDeadline(time.Now().Add(10 * time.Second)); err != nil {
		t.Fatal(err)
	}
	one := make([]byte, 1)
	if _, err := io.ReadFull(conn, one); err != nil {
		t.Fatalf("establish stream: %v", err)
	}

	if err := rig.leases.Release("uuid-head", grant.LeaseID); err != nil {
		t.Fatalf("Release: %v", err)
	}

	if err := conn.SetReadDeadline(time.Now().Add(10 * time.Second)); err != nil {
		t.Fatal(err)
	}
	if _, err := conn.Read(one); err == nil {
		t.Fatal("the stream survived its lease being released")
	}
}

// A head asking a donor that cannot serve deserves a reason, not a 404 it will
// read as a bad address.
func TestANodeWithoutPoolingSaysSo(t *testing.T) {
	root := t.TempDir()
	headDir := filepath.Join(root, "head")
	donorDir := filepath.Join(root, "donor")
	clustertrusttest.Pair(t, "cluster-1", headDir, donorDir, "uuid-head", "uuid-donor")

	collector := NewCollector(fakeNodeInfo(t, twoGPUInventory), donorEnabled(t, 0), nil, "donor")
	addr := startPoolServer(t, clustertrust.Open(donorDir), collector, nil)

	headMesh := clustertrust.Open(headDir)
	cfg, ok := headMesh.ClientTLSConfig("uuid-donor")
	if !ok {
		t.Fatal("no client config")
	}
	client := &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{TLSClientConfig: cfg}}
	resp, err := client.Post("https://"+addr+poolwire.LeasesPath, "application/json", bytes.NewReader([]byte(`{"poolId":"p","deviceIndexes":[0],"bytes":1}`)))
	if err != nil {
		t.Fatalf("lease request: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("lease request to a non-pooling node got %s, want 503", resp.Status)
	}
}

// Lending memory is a decision. A node that has not made it refuses, even to a
// peer it trusts completely.
func TestANonDonorRefusesLeases(t *testing.T) {
	root := t.TempDir()
	headDir := filepath.Join(root, "head")
	donorDir := filepath.Join(root, "donor")
	clustertrusttest.Pair(t, "cluster-1", headDir, donorDir, "uuid-head", "uuid-donor")

	target := newEchoTarget(t)
	leases := NewLeaseStore(target)
	collector := NewCollector(fakeNodeInfo(t, twoGPUInventory), &DonorState{}, leases, "donor")
	addr := startPoolServer(t, clustertrust.Open(donorDir), collector, leases)

	cfg, ok := clustertrust.Open(headDir).ClientTLSConfig("uuid-donor")
	if !ok {
		t.Fatal("no client config")
	}
	client := &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{TLSClientConfig: cfg}}
	resp, err := client.Post("https://"+addr+poolwire.LeasesPath, "application/json", bytes.NewReader([]byte(`{"poolId":"p","deviceIndexes":[0],"bytes":1}`)))
	if err != nil {
		t.Fatalf("lease request: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("lease request to a non-donor got %s, want 403", resp.Status)
	}
}

func TestLeaseRenewAndReleaseOverHTTP(t *testing.T) {
	rig := newDonorRig(t)
	grant := rig.requestLease(t, "pool-1", []int{0}, gb)
	client := rig.headClient(t)
	url := "https://" + rig.donorAddr + poolwire.LeasePathPrefix + grant.LeaseID

	resp, err := client.Post(url, "application/json", bytes.NewReader([]byte(`{"ttlMs":120000}`)))
	if err != nil {
		t.Fatalf("renew: %v", err)
	}
	var renewed poolwire.LeaseGrant
	if err := json.NewDecoder(resp.Body).Decode(&renewed); err != nil {
		t.Fatalf("decode renewal: %v", err)
	}
	_ = resp.Body.Close()
	if renewed.ExpiresAtMs <= grant.ExpiresAtMs {
		t.Fatal("renewal did not extend the lease")
	}

	req, err := http.NewRequest(http.MethodDelete, url, nil)
	if err != nil {
		t.Fatal(err)
	}
	del, err := client.Do(req)
	if err != nil {
		t.Fatalf("release: %v", err)
	}
	_ = del.Body.Close()
	if del.StatusCode != http.StatusNoContent {
		t.Fatalf("release returned %s, want 204", del.Status)
	}
	if len(rig.leases.Snapshot()) != 0 {
		t.Fatal("the lease survived its release")
	}
}

func TestPathSuffix(t *testing.T) {
	cases := []struct {
		path, prefix, want string
		ok                 bool
	}{
		{"/v1/pool/rpc/abc", "/v1/pool/rpc/", "abc", true},
		{"/v1/pool/rpc/", "/v1/pool/rpc/", "", false},
		{"/v1/pool/rpc/a/b", "/v1/pool/rpc/", "", false},
		{"/v1/pool/rpc", "/v1/pool/rpc/", "", false},
		{"/elsewhere/abc", "/v1/pool/rpc/", "", false},
	}
	for _, tc := range cases {
		got, ok := pathSuffix(tc.path, tc.prefix)
		if got != tc.want || ok != tc.ok {
			t.Errorf("pathSuffix(%q) = (%q, %v), want (%q, %v)", tc.path, got, ok, tc.want, tc.ok)
		}
	}
}

func TestHeaderHasToken(t *testing.T) {
	if !headerHasToken([]string{"nvpair-ggml-rpc"}, poolwire.TunnelProtocol) {
		t.Error("exact token should match")
	}
	if !headerHasToken([]string{"NVPAIR-GGML-RPC"}, poolwire.TunnelProtocol) {
		t.Error("token match should be case-insensitive")
	}
	if !headerHasToken([]string{"websocket, nvpair-ggml-rpc"}, poolwire.TunnelProtocol) {
		t.Error("token in a comma list should match")
	}
	if headerHasToken([]string{"websocket"}, poolwire.TunnelProtocol) {
		t.Error("a different protocol must not match")
	}
	if headerHasToken(nil, poolwire.TunnelProtocol) {
		t.Error("no header must not match")
	}
}

// newBufReader is a tiny helper so the raw-upgrade tests read responses through
// a buffered reader, exactly as the production client does.
func newBufReader(conn net.Conn) *bufio.Reader { return bufio.NewReader(conn) }
