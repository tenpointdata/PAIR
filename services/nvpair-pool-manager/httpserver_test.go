// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"io"
	"net/http"
	"path/filepath"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	"nvpair-shared/poolwire"
)

// startServer runs a pool server on an OS-chosen port and returns its address.
func startServer(t *testing.T, mesh *clustertrust.Mesh, collector *Collector) string {
	t.Helper()
	return startPoolServer(t, mesh, collector, nil)
}

// startPoolServer runs a pool server on an OS-chosen port and returns its
// address.
func startPoolServer(t *testing.T, mesh *clustertrust.Mesh, collector *Collector, leases *LeaseStore) string {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	srv := NewServer(0, mesh, collector, leases)
	errCh := make(chan error, 1)
	go func() { errCh <- srv.Run(ctx) }()

	select {
	case <-srv.Bound():
	case err := <-errCh:
		t.Fatalf("server exited before binding: %v", err)
	case <-time.After(5 * time.Second):
		t.Fatal("server did not bind")
	}
	return srv.Addr()
}

// get performs one request with the given client TLS config.
func get(t *testing.T, cfg *tls.Config, url string) (int, []byte) {
	t.Helper()
	client := &http.Client{
		Timeout:   5 * time.Second,
		Transport: &http.Transport{TLSClientConfig: cfg},
	}
	resp, err := client.Get(url)
	if err != nil {
		return 0, nil
	}
	defer func() { _ = resp.Body.Close() }()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	return resp.StatusCode, body
}

// twoNodes returns a server-side mesh and a client-side mesh that have paired
// with each other, plus a stranger that has not.
func twoNodes(t *testing.T) (self, peer, stranger *clustertrust.Mesh) {
	t.Helper()
	root := t.TempDir()
	selfDir := filepath.Join(root, "self")
	peerDir := filepath.Join(root, "peer")
	strangerDir := filepath.Join(root, "stranger")

	selfNode, _ := clustertrusttest.Pair(t, "cluster-1", selfDir, peerDir, "uuid-self", "uuid-peer")

	// The stranger is in a cluster of its own and pins self, so it can complete
	// a handshake; self does not pin it, so the gate is what must refuse it.
	strangerNode := clustertrusttest.MintNode(t, "uuid-stranger")
	clustertrusttest.WriteNode(t, strangerDir, "cluster-2", strangerNode)
	clustertrusttest.PinNode(t, strangerDir, selfNode)

	return clustertrust.Open(selfDir), clustertrust.Open(peerDir), clustertrust.Open(strangerDir)
}

func mustClientConfig(t *testing.T, m *clustertrust.Mesh, peerUUID string) *tls.Config {
	t.Helper()
	cfg, ok := m.ClientTLSConfig(peerUUID)
	if !ok {
		t.Fatalf("no client config for %s", peerUUID)
	}
	return cfg
}

func TestCapacityIsServedToAPinnedPeer(t *testing.T) {
	self, peer, _ := twoNodes(t)
	collector := NewCollector(fakeNodeInfo(t, twoGPUInventory), donorEnabled(t, gb), nil, "fallback")
	url := "https://" + startServer(t, self, collector) + poolwire.CapacityPath

	code, body := get(t, mustClientConfig(t, peer, "uuid-self"), url)
	if code != http.StatusOK {
		t.Fatalf("pinned peer got %d, want 200", code)
	}
	var got poolwire.NodeCapacity
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("parse capacity: %v (body %s)", err, body)
	}
	if !got.DonorEnabled || len(got.Devices) != 2 {
		t.Fatalf("capacity = %+v, want an enabled donor with two devices", got)
	}
	if got.Devices[0].ReservedBytes != gb {
		t.Fatalf("the owner's reservation did not reach the peer: %+v", got.Devices[0])
	}
}

// A machine that can reach the port and completes a handshake, but that this
// node has not paired with, learns nothing.
func TestCapacityIsRefusedToAnUnpinnedCaller(t *testing.T) {
	self, _, stranger := twoNodes(t)
	collector := NewCollector(fakeNodeInfo(t, twoGPUInventory), donorEnabled(t, 0), nil, "fallback")
	url := "https://" + startServer(t, self, collector) + poolwire.CapacityPath

	code, body := get(t, mustClientConfig(t, stranger, "uuid-self"), url)
	if code != http.StatusForbidden {
		t.Fatalf("unpinned caller got %d, want 403", code)
	}
	if len(body) > 0 && json.Valid(body) {
		var leaked poolwire.NodeCapacity
		if err := json.Unmarshal(body, &leaked); err == nil && len(leaked.Devices) > 0 {
			t.Fatal("the refusal body leaked capacity")
		}
	}
}

// There is no plaintext personality on this port. Everything it serves is
// cluster state, and the same listener will carry the ggml tunnel, whose entire
// safety rests on nothing unpinned being able to open it.
func TestPlainHTTPIsNotServedAtAll(t *testing.T) {
	self, _, _ := twoNodes(t)
	collector := NewCollector(fakeNodeInfo(t, twoGPUInventory), donorEnabled(t, 0), nil, "fallback")
	addr := startServer(t, self, collector)

	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get("http://" + addr + poolwire.CapacityPath)
	if err != nil {
		// The connection was refused outright, which is also a pass.
		return
	}
	defer func() { _ = resp.Body.Close() }()

	// Go's TLS listener answers a plaintext request at the TLS layer, before any
	// handler runs, so what comes back is a 400 rather than a connection error.
	// The guarantee being pinned here is not the status code — it is that no
	// handler on this port ever saw the request.
	if resp.StatusCode == http.StatusOK {
		t.Fatal("plain HTTP was served; this port must be mTLS only")
	}
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	var leaked poolwire.NodeCapacity
	if json.Unmarshal(body, &leaked) == nil && len(leaked.Devices) > 0 {
		t.Fatalf("plain HTTP received capacity: %s", body)
	}
}

// A node that belongs to no cluster has nobody it may legitimately serve, so it
// presents no leaf and every handshake fails. This is the state a fresh install
// is in, and it must not be an open capacity endpoint.
func TestUnclusteredNodeServesNothing(t *testing.T) {
	_, peer, _ := twoNodes(t)
	lonely := clustertrust.Open(t.TempDir())
	collector := NewCollector(fakeNodeInfo(t, twoGPUInventory), donorEnabled(t, 0), nil, "fallback")
	url := "https://" + startServer(t, lonely, collector) + poolwire.CapacityPath

	if code, _ := get(t, mustClientConfig(t, peer, "uuid-self"), url); code != 0 {
		t.Fatalf("an unclustered node answered with %d; it should refuse the handshake", code)
	}
}

// Removing a peer has to take effect on the next request, not the next restart:
// that is the difference between revoking donation and merely intending to.
func TestRemovingAPeerTakesEffectImmediately(t *testing.T) {
	root := t.TempDir()
	selfDir := filepath.Join(root, "self")
	peerDir := filepath.Join(root, "peer")
	clustertrusttest.Pair(t, "cluster-1", selfDir, peerDir, "uuid-self", "uuid-peer")

	self := clustertrust.Open(selfDir)
	peer := clustertrust.Open(peerDir)
	collector := NewCollector(fakeNodeInfo(t, twoGPUInventory), donorEnabled(t, 0), nil, "fallback")
	url := "https://" + startServer(t, self, collector) + poolwire.CapacityPath

	cfg := mustClientConfig(t, peer, "uuid-self")
	if code, _ := get(t, cfg, url); code != http.StatusOK {
		t.Fatalf("pinned peer got %d before removal, want 200", code)
	}

	clustertrusttest.RemovePeerPin(t, selfDir, "uuid-peer")

	// A fresh transport, because a pooled connection was authorized by a
	// handshake that has already happened; the gate runs per request and is what
	// must now refuse.
	if code, _ := get(t, mustClientConfig(t, peer, "uuid-self"), url); code != http.StatusForbidden {
		t.Fatalf("removed peer got %d, want 403", code)
	}
}

func TestCapacityRejectsNonGET(t *testing.T) {
	self, peer, _ := twoNodes(t)
	collector := NewCollector(fakeNodeInfo(t, twoGPUInventory), donorEnabled(t, 0), nil, "fallback")
	url := "https://" + startServer(t, self, collector) + poolwire.CapacityPath

	client := &http.Client{
		Timeout:   5 * time.Second,
		Transport: &http.Transport{TLSClientConfig: mustClientConfig(t, peer, "uuid-self")},
	}
	resp, err := client.Post(url, "application/json", nil)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("POST got %d, want 405", resp.StatusCode)
	}
}
