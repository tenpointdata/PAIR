// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"io"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"nvpair-shared/poolwire"
)

// helperEnv marks the re-executed test binary as the stand-in donor backend.
const helperEnv = "NVPAIR_POOL_TEST_BACKEND"

// poolHelperEnv marks the re-executed test binary as the stand-in pool server.
const poolHelperEnv = "NVPAIR_POOL_TEST_SERVER"

// TestPoolDonorHelper is not a test. It is the donor backend the ExecTarget
// tests run, re-executing this same binary the way os/exec's own tests do, so
// the process lifecycle is exercised against a real child process without
// needing llama.cpp or a second compilation step.
//
// It binds whatever `bind=` says — including, deliberately, a wildcard address,
// which is how the loopback-only check gets something real to catch. The
// argument is positional rather than a flag because this binary's flag set is
// the testing package's, and an unknown dashed flag would abort it before main
// ran.
func TestPoolDonorHelper(t *testing.T) {
	if os.Getenv(helperEnv) == "" {
		t.Skip("not the donor backend helper")
	}
	bind := ""
	for _, arg := range os.Args {
		if v, ok := strings.CutPrefix(arg, "bind="); ok {
			bind = v
		}
	}
	if bind == "" {
		os.Exit(2)
	}
	ln, err := net.Listen("tcp", bind)
	if err != nil {
		os.Exit(3)
	}
	for {
		conn, err := ln.Accept()
		if err != nil {
			os.Exit(0)
		}
		_ = conn.Close()
	}
}

// TestPoolServerHelper is not a test either. It stands in for llama-server: it
// reaches every --rpc endpoint it was given, and only then reports itself
// healthy.
//
// Reaching them is the point. A pool server that merely started proves nothing;
// one that has exchanged bytes with each donor has proved the whole path —
// placeholder expansion, the loopback link, cluster mTLS, and the donor's own
// backend — end to end.
func TestPoolServerHelper(t *testing.T) {
	if os.Getenv(poolHelperEnv) == "" {
		t.Skip("not the pool server helper")
	}
	var serve, rpc string
	for _, arg := range os.Args {
		if v, ok := strings.CutPrefix(arg, "serve="); ok {
			serve = v
		}
		if v, ok := strings.CutPrefix(arg, "rpc="); ok {
			rpc = v
		}
	}
	if serve == "" {
		os.Exit(2)
	}

	healthy := true
	for _, endpoint := range strings.Split(rpc, ",") {
		if endpoint == "" {
			continue
		}
		conn, err := net.DialTimeout("tcp", endpoint, 10*time.Second)
		if err != nil {
			healthy = false
			break
		}
		if _, err := conn.Write([]byte("ping")); err != nil {
			healthy = false
			_ = conn.Close()
			break
		}
		buf := make([]byte, 4)
		_ = conn.SetReadDeadline(time.Now().Add(10 * time.Second))
		if _, err := io.ReadFull(conn, buf); err != nil || string(buf) != "ping" {
			healthy = false
		}
		_ = conn.Close()
		if !healthy {
			break
		}
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		if !healthy {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	})
	// Echoes back the argv it was started with, so a test can assert that every
	// placeholder was expanded rather than passed through.
	mux.HandleFunc("/argv", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(strings.Join(os.Args, " ")))
	})
	srv := &http.Server{Addr: serve, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	_ = srv.ListenAndServe()
	os.Exit(0)
}

// helperCommand builds a donor command template that re-executes this binary.
func helperCommand(t *testing.T, bindTemplate string) []string {
	t.Helper()
	return []string{os.Args[0], "-test.run=TestPoolDonorHelper", "bind=" + bindTemplate}
}

// withHelperEnv makes a target's children run as the helper.
func withHelperEnv(t *testing.T) {
	t.Helper()
	t.Setenv(helperEnv, "1")
}

// shortDonorTunables keeps a failing start from waiting out the production
// ninety seconds.
func shortDonorTunables() donorTunables {
	return donorTunables{readyTimeout: 15 * time.Second, readyPoll: 50 * time.Millisecond, stopGrace: time.Second}
}

// helperTarget builds an ExecTarget with test-scale timings.
func helperTarget(t *testing.T, command []string) *ExecTarget {
	t.Helper()
	target := NewExecTarget(command)
	if target != nil {
		target.tunables = shortDonorTunables()
	}
	return target
}

func TestNoCommandMeansNoTarget(t *testing.T) {
	if NewExecTarget(nil) != nil {
		t.Fatal("a node with no donor command must not get a target")
	}
	if NewExecTarget([]string{}) != nil {
		t.Fatal("an empty command must not get a target")
	}
}

func TestExecTargetStartsABackendOnLoopback(t *testing.T) {
	withHelperEnv(t)
	target := helperTarget(t, helperCommand(t, PlaceholderHost+":"+PlaceholderPort))
	grant := poolwire.LeaseGrant{LeaseID: "lease-1", PoolID: "pool-1", DeviceIndexes: []int{0, 1}}

	addr, err := target.Start(context.Background(), grant)
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { target.Stop(grant.LeaseID) })

	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		t.Fatalf("address %q: %v", addr, err)
	}
	if ip := net.ParseIP(host); ip == nil || !ip.IsLoopback() {
		t.Fatalf("backend address %s is not loopback", addr)
	}

	conn, err := net.DialTimeout("tcp", addr, 5*time.Second)
	if err != nil {
		t.Fatalf("backend not accepting at %s: %v", addr, err)
	}
	_ = conn.Close()
}

// The single most important behavior in this file. A backend that bound a
// wildcard address has put an unauthenticated, memory-unsafe tensor protocol on
// every interface the machine has — and it would work perfectly, so nothing else
// would notice. It is checked rather than trusted.
func TestABackendThatBindsBeyondLoopbackIsKilledAndRefused(t *testing.T) {
	if !hasNonLoopbackAddress(t) {
		t.Skip("host has no non-loopback address to probe from")
	}
	withHelperEnv(t)
	target := helperTarget(t, helperCommand(t, "0.0.0.0:"+PlaceholderPort))
	grant := poolwire.LeaseGrant{LeaseID: "lease-wild", PoolID: "pool-1", DeviceIndexes: []int{0}}

	addr, err := target.Start(context.Background(), grant)
	if err == nil {
		target.Stop(grant.LeaseID)
		t.Fatalf("a wildcard bind was accepted, returning %s", addr)
	}
	if !strings.Contains(err.Error(), "must bind") {
		t.Fatalf("error %q does not explain the loopback requirement", err)
	}

	// The offending process must be gone, not merely unreported.
	target.mu.Lock()
	running := len(target.running)
	target.mu.Unlock()
	if running != 0 {
		t.Fatal("the refused backend is still tracked as running")
	}
}

func TestABackendThatNeverAcceptsFailsTheStart(t *testing.T) {
	withHelperEnv(t)
	// No bind= argument, so the helper exits immediately and nothing listens.
	target := helperTarget(t, []string{os.Args[0], "-test.run=TestPoolDonorHelper"})
	target.tunables.readyTimeout = 2 * time.Second
	_, err := target.Start(context.Background(), poolwire.LeaseGrant{LeaseID: "lease-dead", DeviceIndexes: []int{0}})
	if err == nil {
		t.Fatal("a backend that never accepts should fail the start")
	}
}

func TestStopTerminatesTheBackend(t *testing.T) {
	withHelperEnv(t)
	target := helperTarget(t, helperCommand(t, PlaceholderHost+":"+PlaceholderPort))
	grant := poolwire.LeaseGrant{LeaseID: "lease-1", DeviceIndexes: []int{0}}
	addr, err := target.Start(context.Background(), grant)
	if err != nil {
		t.Fatalf("Start: %v", err)
	}

	target.Stop(grant.LeaseID)

	// Stop waits for the process to be reaped, so the port is free the moment it
	// returns rather than at some point afterwards.
	if conn, err := net.DialTimeout("tcp", addr, time.Second); err == nil {
		_ = conn.Close()
		t.Fatal("the backend is still accepting after Stop")
	}
	// Stopping an unknown lease is a no-op, because teardown paths run more than
	// once.
	target.Stop("never-existed")
}

func TestDevicePlaceholderReachesTheCommand(t *testing.T) {
	withHelperEnv(t)
	// The helper ignores everything but bind=, so this asserts substitution
	// rather than behavior: an unsubstituted placeholder would leave a literal
	// "{devices}" in argv, which the loop below would find.
	target := helperTarget(t, append(helperCommand(t, PlaceholderHost+":"+PlaceholderPort), "devices="+PlaceholderDevices))
	grant := poolwire.LeaseGrant{LeaseID: "lease-1", DeviceIndexes: []int{2, 5}}
	if _, err := target.Start(context.Background(), grant); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { target.Stop(grant.LeaseID) })

	target.mu.Lock()
	proc := target.running[grant.LeaseID]
	target.mu.Unlock()
	if proc == nil {
		t.Fatal("no process recorded")
	}
	joined := strings.Join(proc.cmd.Args, " ")
	if strings.Contains(joined, "{") {
		t.Fatalf("argv still holds an unsubstituted placeholder: %s", joined)
	}
	if !strings.Contains(joined, "2,5") {
		t.Fatalf("argv %s does not carry the device list", joined)
	}
	var sawVisible bool
	for _, kv := range proc.cmd.Env {
		if kv == "CUDA_VISIBLE_DEVICES=2,5" {
			sawVisible = true
		}
	}
	if !sawVisible {
		t.Fatal("the backend was not confined to the leased devices")
	}
}

// The template is expanded and executed directly, so there is no shell to
// interpret a semicolon, a pipe, or a backtick in a value that reached this
// process from a settings file.
func TestSplitCommandQuotingWithoutAShell(t *testing.T) {
	cases := map[string][]string{
		"":                                      nil,
		"rpc-server -H {host} -p {port}":        {"rpc-server", "-H", "{host}", "-p", "{port}"},
		`"/opt/llama cpp/rpc-server" -p {port}`: {"/opt/llama cpp/rpc-server", "-p", "{port}"},
		"  spaced   out  ":                      {"spaced", "out"},
		"a; rm -rf /":                           {"a;", "rm", "-rf", "/"},
	}
	for in, want := range cases {
		got := splitCommand(in)
		if len(got) != len(want) {
			t.Errorf("splitCommand(%q) = %v, want %v", in, got, want)
			continue
		}
		for i := range got {
			if got[i] != want[i] {
				t.Errorf("splitCommand(%q)[%d] = %q, want %q", in, i, got[i], want[i])
			}
		}
	}
}

func TestFreeLoopbackPortReturnsSomethingUsable(t *testing.T) {
	port, err := freeLoopbackPort()
	if err != nil {
		t.Fatalf("freeLoopbackPort: %v", err)
	}
	if port <= 0 || port > 65535 {
		t.Fatalf("port %d is not usable", port)
	}
	ln, err := net.Listen("tcp", net.JoinHostPort(donorHost, strconv.Itoa(port)))
	if err != nil {
		t.Fatalf("the port it chose could not be bound: %v", err)
	}
	_ = ln.Close()
}

// hasNonLoopbackAddress reports whether this host can probe itself from an
// address other than loopback, which the wildcard-bind check needs.
func hasNonLoopbackAddress(t *testing.T) bool {
	t.Helper()
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return false
	}
	for _, a := range addrs {
		if ipnet, ok := a.(*net.IPNet); ok && !ipnet.IP.IsLoopback() && !ipnet.IP.IsLinkLocalUnicast() {
			return true
		}
	}
	return false
}
