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
	"syscall"

	"nvpair-shared/appdir"
	"nvpair-shared/applog"
	"nvpair-shared/clustertrust"
	"nvpair-shared/nodeid"
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

	collector := NewCollector(*nodeInfoURL, donor, nil, nodeid.Resolve(""))
	peers := NewPeerCollector(mesh)

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

	server := NewServer(*port, mesh, collector)
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
	mgr := NewManager(codec, donor, collector, peers)
	if err := mgr.Run(ctx); err != nil && ctx.Err() == nil {
		log.Fatalf("manager error: %v", err)
	}
	slog.Info("shutdown complete")
}
