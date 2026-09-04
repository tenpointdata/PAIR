// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/poolwire"
)

// leaseCallTimeout bounds one lease call. Generous because granting a lease
// starts a backend process on the far side, and a cold CUDA context is slow —
// the donor's own readiness timeout is the real bound, and this must not expire
// first and leave a backend running that this node believes failed.
const leaseCallTimeout = 2 * time.Minute

// LeaseClient is the head's side of the lease protocol.
type LeaseClient struct {
	mesh    *clustertrust.Mesh
	clients *clustertrust.PeerClientPool
}

func NewLeaseClient(mesh *clustertrust.Mesh) *LeaseClient {
	return &LeaseClient{mesh: mesh, clients: clustertrust.NewPeerClientPool(mesh, leaseCallTimeout)}
}

// Request asks a donor to hold memory for a pool.
func (c *LeaseClient) Request(ctx context.Context, peerUUID, addr string, req poolwire.LeaseRequest) (poolwire.LeaseGrant, error) {
	var grant poolwire.LeaseGrant
	body, err := json.Marshal(req)
	if err != nil {
		return grant, fmt.Errorf("encode lease request: %w", err)
	}
	resp, err := c.do(ctx, peerUUID, http.MethodPost, "https://"+addr+poolwire.LeasesPath, body)
	if err != nil {
		return grant, err
	}
	defer closeBody(resp)

	if resp.StatusCode != http.StatusOK {
		// The donor's refusal text is the useful part — "device 0 has 6 GiB
		// free, needs 14" is what a user can act on — so it is carried up rather
		// than replaced with a status code.
		return grant, fmt.Errorf("donor refused: %s", statusDetail(resp))
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, leaseMaxBytes)).Decode(&grant); err != nil {
		return grant, fmt.Errorf("decode grant: %w", err)
	}
	return grant, nil
}

// Renew extends a lease.
func (c *LeaseClient) Renew(ctx context.Context, peerUUID, addr, leaseID string, ttl time.Duration) (poolwire.LeaseGrant, error) {
	var grant poolwire.LeaseGrant
	body, err := json.Marshal(poolwire.LeaseRequest{TTLMs: ttl.Milliseconds()})
	if err != nil {
		return grant, fmt.Errorf("encode renewal: %w", err)
	}
	resp, err := c.do(ctx, peerUUID, http.MethodPost, "https://"+addr+poolwire.LeasePathPrefix+leaseID, body)
	if err != nil {
		return grant, err
	}
	defer closeBody(resp)

	if resp.StatusCode != http.StatusOK {
		return grant, fmt.Errorf("renewal refused: %s", statusDetail(resp))
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, leaseMaxBytes)).Decode(&grant); err != nil {
		return grant, fmt.Errorf("decode renewal: %w", err)
	}
	return grant, nil
}

// Release ends a lease. A lease the donor no longer has is success: teardown
// paths run more than once, and a lease that lapsed on its own has already done
// what releasing it would do.
func (c *LeaseClient) Release(ctx context.Context, peerUUID, addr, leaseID string) error {
	resp, err := c.do(ctx, peerUUID, http.MethodDelete, "https://"+addr+poolwire.LeasePathPrefix+leaseID, nil)
	if err != nil {
		return err
	}
	defer closeBody(resp)

	switch resp.StatusCode {
	case http.StatusNoContent, http.StatusNotFound:
		return nil
	default:
		return fmt.Errorf("release refused: %s", statusDetail(resp))
	}
}

func (c *LeaseClient) do(ctx context.Context, peerUUID, method, url string, body []byte) (*http.Response, error) {
	client, ok := c.clients.Client(peerUUID)
	if !ok {
		return nil, fmt.Errorf("no pin for donor %s", peerUUID)
	}
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, reader)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("call donor: %w", err)
	}
	return resp, nil
}

// statusDetail renders a failing response as status plus whatever explanation it
// carried.
func statusDetail(resp *http.Response) string {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
	detail := string(bytes.TrimSpace(body))
	if detail == "" {
		return resp.Status
	}
	return resp.Status + ": " + detail
}

func closeBody(resp *http.Response) {
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
}
