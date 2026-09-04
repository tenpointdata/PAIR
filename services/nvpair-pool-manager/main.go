// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"nvpair-shared/appdir"
	"nvpair-shared/applog"
	"nvpair-shared/clustertrust"
	"nvpair-shared/nodeid"
	"nvpair-shared/vrampool"
)

// DefaultPort is the cluster-mTLS port this service binds. Fixed, like
// node-info's :14318 and engine-manager's :14322, and the next free number in
// the 143xx range — the broker knows it, so no dynamic port handshake is needed.
const DefaultPort = 14324

// defaultNodeInfoURL is the local hardware inventory. Loopback because this is
// the same machine: reading our own inventory over the LAN would work, and would
// also mean a firewall rule could stop a node from knowing its own GPUs.
const defaultNodeInfoURL = "http://127.0.0.1:14318/v1/node-info"

// defaultDonorPath returns the canonical on-disk location for the donation
// policy, alongside the other per-user PAIR configuration.
func defaultDonorPath() string {
	if p, err := appdir.Path("pool-donor.json"); err == nil {
		return p
	}
	if exe, err := os.Executable(); err == nil {
		return filepath.Join(filepath.Dir(exe), "pool-donor.json")
	}
	return "pool-donor.json"
}

func main() {
	ipcPath := flag.String("ipc", "", "IPC endpoint: Unix domain socket path or Windows named pipe (default: stdin/stdout)")
	showVersion := flag.Bool("version", false, "print version and exit")
	port := flag.Int("port", DefaultPort, "cluster-mTLS port for the pool surface")
	clusterDir := flag.String("cluster-dir", "", "cluster config dir; peers are authenticated against the pins it holds")
	nodeInfoURL := flag.String("node-info-url", defaultNodeInfoURL, "local nvpair-node-info inventory URL")
	donorPath := flag.String("donor-settings", "", "path to pool-donor.json (default: the per-user PAIR data dir)")
	rpcCommand := flag.String("donor-command", "", "command template that runs a ggml RPC backend for one lease, e.g. \"/path/to/rpc-server -H {host} -p {port}\". It MUST bind {host}; a backend found listening anywhere else is killed and the lease refused. Without it this node cannot donate.")
	headCommand := flag.String("head-command", "", "command template that runs a pool's server, e.g. \"/path/to/llama-server -m {model} --host {host} --port {port} --rpc {rpc} --tensor-split {split} -c {context} -ngl 999\". Without it this node cannot head a pool.")
	allowWiFiDonors := flag.Bool("allow-wifi-donors", false, "let a pool use a donor reached over a wireless link. Off by default: a wireless hop is slow for every token and is the one most likely to drop, and losing a donor fails the whole pool.")
	poolHeadroom := flag.Float64("pool-headroom", 0, "fraction of a device's free VRAM a pool may claim (0 uses the built-in default)")
	resolveLevel := applog.RegisterFlag(nil, slog.LevelInfo)
	flag.Parse()

	if *showVersion {
		fmt.Println(Version)
		os.Exit(0)
	}

	applog.Init("nvpair-pool-manager", resolveLevel())

	var transport io.ReadWriteCloser
	if *ipcPath != "" {
		conn, err := dialIPC(*ipcPath)
		if err != nil {
			log.Fatalf("failed to connect to IPC endpoint %q: %v", *ipcPath, err)
		}
		transport = conn
		slog.Info("using IPC transport", "path", *ipcPath)
	} else {
		transport = newStdioTransport()
		slog.Info("using stdio transport")
	}
	defer func() { _ = transport.Close() }()

	path := *donorPath
	if path == "" {
		path = defaultDonorPath()
	}
	donor, err := LoadDonorState(path)
	if err != nil {
		log.Fatalf("failed to load donor settings: %v", err)
	}

	// A live mesh rather than a snapshot: this node can pair, or leave a
	// cluster, long after this process starts, and the listener has to follow it
	// without a restart. An empty cluster dir yields a mesh that is permanently
	// unclustered, which for this service means a port that refuses every
	// handshake — the correct behavior for a node with nobody to trust.
	mesh := clustertrust.Open(*clusterDir)

	// A node with no donor command cannot lend memory, so it gets no lease
	// store and its lease endpoints say so. That is the honest shape: a donor
	// with nothing to run is not a donor, and pretending otherwise would have a
	// head plan a pool around memory that can never answer.
	var leases *LeaseStore
	if target := NewExecTarget(splitCommand(*rpcCommand)); target != nil {
		leases = NewLeaseStore(target)
		defer leases.ReleaseAll()
	} else if *rpcCommand != "" {
		log.Fatalf("--donor-command %q is not a runnable command", *rpcCommand)
	}

	collector := NewCollector(*nodeInfoURL, donor, leases, nodeid.Resolve(""))
	peers := NewPeerCollector(mesh)

	// Heading a pool and donating to one are separate capabilities. A machine
	// can lend its GPU without having a server build, and a laptop with a server
	// build may have nothing worth lending, so the two commands are configured
	// and reported independently.
	pools := NewPoolManager(mesh, peers, collector, splitCommand(*headCommand), vrampool.Policy{
		Headroom:        *poolHeadroom,
		AllowWiFiDonors: *allowWiFiDonors,
	})
	defer pools.TeardownAll()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		select {
		case sig := <-sigCh:
			slog.Info("received signal, shutting down", "signal", sig.String())
			cancel()
		case <-ctx.Done():
		}
	}()

	if leases != nil {
		go sweepLeases(ctx, leases)
	}

	server := NewServer(*port, mesh, collector, leases)
	go func() {
		if err := server.Run(ctx); err != nil {
			// A bind failure is not fatal to the control channel. The broker can
			// still read this node's own capacity and change its donation
			// policy; what is lost is peers being able to ask, which is a
			// degraded service rather than a dead one, and is worth reporting
			// rather than exiting over.
			slog.Error("pool server stopped", "err", err)
		}
	}()

	codec := NewCodec(transport)
	mgr := NewManager(codec, donor, collector, peers, leases, pools)
	if err := mgr.Run(ctx); err != nil && ctx.Err() == nil {
		log.Fatalf("manager error: %v", err)
	}
	slog.Info("shutdown complete")
}

// leaseSweepInterval is how often lapsed leases are reaped.
//
// Well under the minimum lease TTL, so a head that stops renewing gets its
// memory returned promptly rather than at the next time somebody happens to ask
// about capacity. The capacity path sweeps on read as well; this timer is what
// covers a node nobody is asking.
const leaseSweepInterval = 5 * time.Second

// sweepLeases reaps expired leases until ctx is cancelled.
func sweepLeases(ctx context.Context, leases *LeaseStore) {
	ticker := time.NewTicker(leaseSweepInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			leases.Sweep()
		}
	}
}

// splitCommand splits a command template into argv on whitespace, honoring
// double quotes so a path with a space survives.
//
// Deliberately not a shell: the template is expanded and executed directly, so
// there is no shell to interpret a semicolon, a pipe, or a backtick in a value
// that reached this process over a settings file.
func splitCommand(s string) []string {
	var out []string
	var cur strings.Builder
	inQuote := false
	flush := func() {
		if cur.Len() > 0 {
			out = append(out, cur.String())
			cur.Reset()
		}
	}
	for _, r := range s {
		switch {
		case r == '"':
			inQuote = !inQuote
		case !inQuote && (r == ' ' || r == '\t'):
			flush()
		default:
			cur.WriteRune(r)
		}
	}
	flush()
	return out
}
