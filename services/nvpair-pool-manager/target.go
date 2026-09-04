// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"nvpair-shared/poolwire"
)

// Placeholders a donor command template may use, following the same {token}
// convention nvpair-engine-manager's manifests already use so an operator writing
// one is not learning a second syntax.
const (
	// PlaceholderHost is always 127.0.0.1. It is a placeholder rather than a
	// literal so the template reads as the address it will bind, and so the
	// loopback rule below is visible at the point the command is written.
	PlaceholderHost = "{host}"
	// PlaceholderPort is the port this service chose for the lease.
	PlaceholderPort = "{port}"
	// PlaceholderDevices is the comma-separated device indexes the lease covers.
	PlaceholderDevices = "{devices}"
)

// donorHost is the only address a donor backend may bind.
const donorHost = "127.0.0.1"

// donorTunables are the durations a donor backend's lifecycle runs on. Fields
// rather than package-level variables, for the reason poolTunables gives: a
// global that a test shortens is a global that Stop, called from the lease
// sweeper's goroutine, then races over.
type donorTunables struct {
	// readyTimeout bounds how long a backend has to start accepting. A cold
	// CUDA context on a large card is slow, and failing early would refuse
	// leases on exactly the hardware pooling exists for.
	readyTimeout time.Duration
	// readyPoll is how often readiness is retried.
	readyPoll time.Duration
	// stopGrace is how long a backend gets to exit on its own before it is
	// killed.
	stopGrace time.Duration
}

func defaultDonorTunables() donorTunables {
	return donorTunables{
		readyTimeout: 90 * time.Second,
		readyPoll:    200 * time.Millisecond,
		stopGrace:    5 * time.Second,
	}
}

// exposureProbeTimeout bounds one connect attempt in the loopback-only check.
const exposureProbeTimeout = 500 * time.Millisecond

// ExecTarget runs one donor backend process per lease — in practice llama.cpp's
// `rpc-server` — bound to loopback and nothing else.
//
// The command is a template supplied by the operator rather than a compiled-in
// path, for the same reason nvpair-engine-manager drives engines from manifests:
// which llama.cpp build a machine has, and where, is not something PAIR can know,
// and hardcoding it makes the feature unavailable to everyone whose layout
// differs.
type ExecTarget struct {
	// command is the argv template. The first element is the program.
	command  []string
	tunables donorTunables

	mu      sync.Mutex
	running map[string]*donorProcess
}

// donorProcess is one lease's backend.
type donorProcess struct {
	cmd  *exec.Cmd
	addr string
	// done closes when the process has been reaped, so Stop can wait for the
	// port to actually be free rather than for a signal to have been sent.
	done chan struct{}
}

// NewExecTarget returns a target that runs command per lease, or nil when no
// command was configured — a node with no llama.cpp build cannot be a donor, and
// saying so by having no target is clearer than having one that always fails.
func NewExecTarget(command []string) *ExecTarget {
	if len(command) == 0 {
		return nil
	}
	return &ExecTarget{command: command, tunables: defaultDonorTunables(), running: make(map[string]*donorProcess)}
}

// Start brings up the backend for a lease and returns its loopback address.
func (t *ExecTarget) Start(ctx context.Context, grant poolwire.LeaseGrant) (string, error) {
	port, err := freeLoopbackPort()
	if err != nil {
		return "", fmt.Errorf("reserve donor port: %w", err)
	}
	addr := net.JoinHostPort(donorHost, strconv.Itoa(port))

	devices := make([]string, len(grant.DeviceIndexes))
	for i, idx := range grant.DeviceIndexes {
		devices[i] = strconv.Itoa(idx)
	}
	deviceList := strings.Join(devices, ",")

	argv := make([]string, len(t.command))
	for i, arg := range t.command {
		arg = strings.ReplaceAll(arg, PlaceholderHost, donorHost)
		arg = strings.ReplaceAll(arg, PlaceholderPort, strconv.Itoa(port))
		arg = strings.ReplaceAll(arg, PlaceholderDevices, deviceList)
		argv[i] = arg
	}

	// Process context, not the request's: the backend outlives the HTTP call
	// that created it, and tying it to the request would kill it the moment the
	// grant response was written.
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Env = append(os.Environ(),
		// The conventional way to confine a CUDA process to specific cards. It
		// is set alongside {devices} rather than instead of it, because backends
		// differ in which one they honor and setting both is harmless.
		"CUDA_VISIBLE_DEVICES="+deviceList,
		"HIP_VISIBLE_DEVICES="+deviceList,
	)
	// Never inherit stdout: a backend that writes to it would interleave with
	// this process's JSON-RPC frames and corrupt the control channel.
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("start donor backend: %w", err)
	}

	proc := &donorProcess{cmd: cmd, addr: addr, done: make(chan struct{})}
	go func() {
		_ = cmd.Wait()
		close(proc.done)
	}()

	if err := waitForAccept(ctx, addr, t.tunables); err != nil {
		t.kill(proc)
		return "", fmt.Errorf("donor backend never accepted on %s: %w", addr, err)
	}

	// The most important line in this file. A backend that bound a wildcard
	// address has put an unauthenticated, memory-unsafe tensor protocol on every
	// interface this machine has — the exact exposure the whole tunnel exists to
	// prevent — and it would work perfectly, so nothing else would ever notice.
	// A template that binds the wrong address is an easy mistake to make and an
	// impossible one to see, so it is checked rather than trusted.
	if reachable, via := reachableOffLoopback(port); reachable {
		t.kill(proc)
		return "", fmt.Errorf("donor backend on port %d is reachable at %s; it must bind %s only", port, via, donorHost)
	}

	t.mu.Lock()
	t.running[grant.LeaseID] = proc
	t.mu.Unlock()

	slog.Info("donor backend started", "lease", grant.LeaseID, "addr", addr,
		"devices", deviceList, "pid", cmd.Process.Pid)
	return addr, nil
}

// Stop terminates a lease's backend.
func (t *ExecTarget) Stop(leaseID string) {
	t.mu.Lock()
	proc, ok := t.running[leaseID]
	delete(t.running, leaseID)
	t.mu.Unlock()
	if !ok {
		return
	}
	t.kill(proc)
	slog.Info("donor backend stopped", "lease", leaseID)
}

// kill asks a backend to exit, then insists.
//
// The polite signal first because a backend killed mid-allocation can leave a
// driver context that outlives it; the insistence afterwards because a wedged
// backend must not hold a card hostage until the machine reboots.
func (t *ExecTarget) kill(proc *donorProcess) {
	if proc.cmd.Process == nil {
		return
	}
	_ = proc.cmd.Process.Signal(os.Interrupt)
	select {
	case <-proc.done:
		return
	case <-time.After(t.tunables.stopGrace):
	}
	_ = proc.cmd.Process.Kill()
	<-proc.done
}

// freeLoopbackPort asks the OS for an unused port by binding and releasing one.
//
// There is a race between releasing it and the backend claiming it, and it is
// accepted deliberately: the alternative is a fixed port, which cannot support
// two leases on one machine and collides with whatever else chose the same
// number. A lost race fails the lease with a clear "never accepted" error, which
// the head retries.
func freeLoopbackPort() (int, error) {
	ln, err := net.Listen("tcp", donorHost+":0")
	if err != nil {
		return 0, err
	}
	defer func() { _ = ln.Close() }()
	addr, ok := ln.Addr().(*net.TCPAddr)
	if !ok {
		return 0, fmt.Errorf("unexpected listener address %T", ln.Addr())
	}
	return addr.Port, nil
}

// waitForAccept blocks until addr completes a TCP handshake or the deadline
// passes.
func waitForAccept(ctx context.Context, addr string, t donorTunables) error {
	deadline := time.Now().Add(t.readyTimeout)
	for {
		conn, err := net.DialTimeout("tcp", addr, t.readyPoll)
		if err == nil {
			_ = conn.Close()
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("timed out after %s", t.readyTimeout)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(t.readyPoll):
		}
	}
}

// reachableOffLoopback reports whether port answers on any of this host's
// non-loopback addresses, which would mean the backend bound more than it was
// told to.
//
// It probes from this machine, so it sees a wildcard bind even on a host whose
// firewall would drop the same connection from outside. That is the right test:
// a firewall is somebody else's configuration and can change, while what the
// process bound is a fact about the process.
func reachableOffLoopback(port int) (bool, string) {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return false, ""
	}
	for _, a := range addrs {
		ipnet, ok := a.(*net.IPNet)
		if !ok || ipnet.IP.IsLoopback() || ipnet.IP.IsLinkLocalUnicast() {
			continue
		}
		target := net.JoinHostPort(ipnet.IP.String(), strconv.Itoa(port))
		conn, err := net.DialTimeout("tcp", target, exposureProbeTimeout)
		if err != nil {
			continue
		}
		_ = conn.Close()
		return true, target
	}
	return false, ""
}
