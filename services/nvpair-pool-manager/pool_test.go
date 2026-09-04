// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	"nvpair-shared/linkq"
	"nvpair-shared/poolwire"
	"nvpair-shared/vrampool"
)

const mib = 1 << 20

// exactPolicy plans against whole free bytes, so a test can express capacity in
// whole megabytes instead of working backwards through headroom and the compute
// buffer. Those defaults are tested on their own in shared/vrampool.
func exactPolicy() vrampool.Policy {
	return vrampool.Policy{Headroom: 1, PerDeviceOverheadBytes: 1}
}

// inventory renders a node-info body with one GPU of the given size.
func inventory(hostUUID string, totalBytes, usedBytes uint64) string {
	return fmt.Sprintf(
		`{"GPUs":[{"name":"Test GPU","vram_bytes":%d,"vram_used_bytes":%d}],"telemetryValid":true,"hostUuid":%q}`,
		totalBytes, usedBytes, hostUUID)
}

// poolModel writes a synthetic GGUF sized so it cannot fit on one small device:
// 4 MiB of non-repeating tensors plus eight 2 MiB blocks.
func poolModel(t *testing.T, dir string) string {
	t.Helper()
	b := &ggufBuilder{}
	b.stringKV("general.architecture", "llama")
	b.uint32KV("llama.block_count", 8)
	b.uint32KV("llama.embedding_length", 1024)
	b.uint32KV("llama.attention.head_count", 8)
	b.uint32KV("llama.attention.head_count_kv", 8)

	b.tensor("token_embd.weight", 0)
	for i := range 8 {
		b.tensor(fmt.Sprintf("blk.%d.attn_q.weight", i), uint64(2*mib+i*2*mib))
	}
	b.tensor("output.weight", uint64(18*mib))

	path := filepath.Join(dir, "pooled-model.gguf")
	b.write(t, path, 20*mib)
	return path
}

// poolRig is a head and a donor, paired, with the donor serving capacity and
// leases for real.
type poolRig struct {
	head      *PoolManager
	peers     *PeerCollector
	donorAddr string
	modelPath string
}

// headCommandTemplate runs the pool-server helper with every placeholder the
// production template supports, so expansion is exercised rather than assumed.
func headCommandTemplate(t *testing.T) []string {
	t.Helper()
	return []string{
		os.Args[0], "-test.run=TestPoolServerHelper",
		"serve=" + PlaceholderHost + ":" + PlaceholderPort,
		"rpc=" + PlaceholderRPC,
		"split=" + PlaceholderSplit,
		"model=" + PlaceholderModel,
		"context=" + PlaceholderContext,
	}
}

// newPoolRig builds a head and a donor with the given per-device VRAM. Most
// tests give both 16 MiB against a model that needs 24 MiB at this context
// length: neither device can hold it alone, which is the only situation pooling
// is for, and together they can.
func newPoolRig(t *testing.T, headVRAM, donorVRAM uint64) *poolRig {
	t.Helper()
	t.Setenv(poolHelperEnv, "1")

	root := t.TempDir()
	headDir := filepath.Join(root, "head")
	donorDir := filepath.Join(root, "donor")
	clustertrusttest.Pair(t, "cluster-1", headDir, donorDir, "uuid-head", "uuid-donor")

	// Donor: a real pool server with a real lease store and an echo backend.
	donorLeases := NewLeaseStore(newEchoTarget(t))
	donorCollector := NewCollector(fakeNodeInfo(t, inventory("uuid-donor", donorVRAM, 0)), donorEnabled(t, 0), donorLeases, "uuid-donor")
	donorAddr := startPoolServer(t, clustertrust.Open(donorDir), donorCollector, donorLeases)

	// Head.
	headMesh := clustertrust.Open(headDir)
	headCollector := NewCollector(fakeNodeInfo(t, inventory("uuid-head", headVRAM, 0)), donorEnabled(t, 0), nil, "uuid-head")
	peers := NewPeerCollector(headMesh)
	peers.SetPeers([]poolwire.Peer{{NodeUUID: "uuid-donor", NodeName: "donor", Addresses: []string{donorAddr}}})
	if got := peers.Collect(context.Background()); len(got) != 1 {
		t.Fatalf("head could not collect the donor's capacity: %+v", got)
	}

	head := NewPoolManager(headMesh, peers, headCollector, headCommandTemplate(t), exactPolicy())
	head.tunables = shortTunables()
	t.Cleanup(head.TeardownAll)

	return &poolRig{head: head, peers: peers, donorAddr: donorAddr, modelPath: poolModel(t, root)}
}

// shortTunables keeps a failing formation from waiting out the production
// forty-five minutes. Set on the manager, not on a global, so the renewal
// goroutine reads a value nothing will change under it.
func shortTunables() poolTunables {
	return poolTunables{
		readyTimeout:  30 * time.Second,
		readyPoll:     100 * time.Millisecond,
		stopGrace:     time.Second,
		renewInterval: 200 * time.Millisecond,
	}
}

// A capacity sweep is a timed request that was going to happen anyway, so the
// planner's donor gate ends up operating on measured evidence rather than on an
// assumption that every peer is equal.
func TestPeerCapacityMeasuresTheLink(t *testing.T) {
	rig := newPoolRig(t, 16*mib, 16*mib)
	path, ok := rig.peers.Path("uuid-donor")
	if !ok {
		t.Fatal("no link measurement after a capacity sweep")
	}
	if path.RTT <= 0 {
		t.Fatal("the sweep should have produced a round-trip time")
	}
	if path.Stability != 1 {
		t.Fatalf("stability = %v after a clean sweep, want 1", path.Stability)
	}
	if path.Class == linkq.ClassUnknown {
		t.Fatal("the path should have been classified")
	}
	// Loopback in a test, which is exactly what it is.
	if path.Class != linkq.ClassLocal {
		t.Fatalf("class = %v, want local for a peer reached over loopback", path.Class)
	}
}

// A model that fits on one node should be ROUTED there, not pooled: routing is
// faster, survives that node going away, and is what PAIR already does.
func TestPlanReportsWhenPoolingIsUnnecessary(t *testing.T) {
	rig := newPoolRig(t, 64*mib, 64*mib)
	got, err := rig.head.Plan(context.Background(), rig.modelPath, 128)
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if !got.FitsOnOneNode {
		t.Fatal("a model that fits on one device should say so")
	}
}

func TestPlanSpreadsAModelThatFitsNowhereAlone(t *testing.T) {
	rig := newPoolRig(t, 16*mib, 16*mib)
	got, err := rig.head.Plan(context.Background(), rig.modelPath, 128)
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if got.FitsOnOneNode {
		t.Fatal("this model should not fit on either device alone")
	}
	if !got.Fits {
		t.Fatalf("plan did not fit: %.2f MiB short, rejected %+v",
			float64(got.ShortfallBytes)/mib, got.Rejected)
	}
	if len(got.Members) != 2 {
		t.Fatalf("members = %+v, want the head and one donor", got.Members)
	}
	if !got.Members[0].IsHead || got.Members[0].NodeUUID != "uuid-head" {
		t.Fatalf("first member = %+v, want the head node", got.Members[0])
	}
	if got.Boundaries != 1 {
		t.Fatalf("boundaries = %d, want 1 for two machines", got.Boundaries)
	}
	total := 0
	for _, mem := range got.Members {
		total += mem.Layers
	}
	if total != 8 {
		t.Fatalf("placed %d layers, want all 8", total)
	}
}

func TestPlanExplainsAModelThatCannotFitAnywhere(t *testing.T) {
	rig := newPoolRig(t, 6*mib, 6*mib)
	got, err := rig.head.Plan(context.Background(), rig.modelPath, 128)
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if got.Fits {
		t.Fatal("a 20 MiB model should not fit in 12 MiB")
	}
	if got.ShortfallBytes == 0 {
		t.Fatal("a non-fit should estimate how much more was needed")
	}
}

// The headline test: a real pool, across two machines, with the server proving
// it reached the donor through the tunnel before reporting itself healthy.
func TestFormBringsUpAPoolAcrossTwoMachines(t *testing.T) {
	rig := newPoolRig(t, 16*mib, 16*mib)

	pool, err := rig.head.Form(context.Background(), "pool-1", rig.modelPath, 128)
	if err != nil {
		t.Fatalf("Form: %v", err)
	}
	t.Cleanup(func() { _ = rig.head.Teardown("pool-1") })

	if pool.Endpoint == "" {
		t.Fatal("a formed pool should expose an endpoint")
	}
	if len(pool.Members) != 2 {
		t.Fatalf("members = %+v, want two", pool.Members)
	}
	if pool.Boundaries != 1 {
		t.Fatalf("boundaries = %d, want 1", pool.Boundaries)
	}

	// The helper only reports healthy after exchanging bytes with every --rpc
	// endpoint, so a 200 here means the whole path worked: placeholder
	// expansion, the loopback link, cluster mTLS, and the donor's backend.
	resp, err := http.Get(pool.Endpoint + "/health")
	if err != nil {
		t.Fatalf("pool endpoint: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("pool health = %s; the server never reached its donors", resp.Status)
	}

	// Every placeholder must have been expanded; a literal brace in argv would
	// mean the server was started with a template rather than a command.
	argvResp, err := http.Get(pool.Endpoint + "/argv")
	if err != nil {
		t.Fatalf("read argv: %v", err)
	}
	defer func() { _ = argvResp.Body.Close() }()
	argv, err := io.ReadAll(io.LimitReader(argvResp.Body, 1<<16))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(argv), "{") {
		t.Fatalf("argv still holds a placeholder: %s", argv)
	}
	for _, want := range []string{"split=", "model=" + rig.modelPath, "context=128"} {
		if !strings.Contains(string(argv), want) {
			t.Errorf("argv %s is missing %q", argv, want)
		}
	}
}

// The donor's memory must be committed for as long as the pool exists, and
// returned when it ends.
func TestFormAndTeardownMoveTheDonorsCommittedMemory(t *testing.T) {
	rig := newPoolRig(t, 16*mib, 16*mib)

	if _, err := rig.head.Form(context.Background(), "pool-1", rig.modelPath, 128); err != nil {
		t.Fatalf("Form: %v", err)
	}

	before := rig.peers.Collect(context.Background())["uuid-donor"]
	if before.Devices[0].CommittedBytes == 0 {
		t.Fatal("the donor should be holding memory for the pool")
	}

	if err := rig.head.Teardown("pool-1"); err != nil {
		t.Fatalf("Teardown: %v", err)
	}
	after := rig.peers.Collect(context.Background())["uuid-donor"]
	if after.Devices[0].CommittedBytes != 0 {
		t.Fatalf("the donor still holds %d bytes after teardown", after.Devices[0].CommittedBytes)
	}
	if len(rig.head.Pools()) != 0 {
		t.Fatal("the pool survived its teardown")
	}
}

// A donor whose lease lapses takes the pool with it, so the head renews for as
// long as the pool lives.
func TestTheHeadKeepsItsLeasesAlive(t *testing.T) {
	rig := newPoolRig(t, 16*mib, 16*mib)

	if _, err := rig.head.Form(context.Background(), "pool-1", rig.modelPath, 128); err != nil {
		t.Fatalf("Form: %v", err)
	}
	t.Cleanup(func() { _ = rig.head.Teardown("pool-1") })

	// Well past several renewal intervals; the lease TTL is far longer, so this
	// only proves the loop is running, which is the part that can silently stop.
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		capacity := rig.peers.Collect(context.Background())["uuid-donor"]
		if len(capacity.Devices) == 0 || capacity.Devices[0].CommittedBytes == 0 {
			t.Fatal("the donor stopped holding memory while the pool was alive")
		}
		time.Sleep(300 * time.Millisecond)
	}
}

// A half-formed pool is worse than none: donors hold memory for a head that is
// not coming, and it only returns when their leases lapse.
func TestAFailedFormationReleasesEveryDonor(t *testing.T) {
	rig := newPoolRig(t, 16*mib, 16*mib)
	rig.head.tunables.readyTimeout = 2 * time.Second

	// A server command that starts and never serves: leases are taken, then
	// readiness fails.
	rig.head.headCommand = []string{os.Args[0], "-test.run=TestPoolServerHelper"}

	if _, err := rig.head.Form(context.Background(), "pool-1", rig.modelPath, 128); err == nil {
		t.Fatal("formation should have failed")
	}
	capacity := rig.peers.Collect(context.Background())["uuid-donor"]
	if len(capacity.Devices) == 0 {
		t.Fatal("no donor capacity reported")
	}
	if capacity.Devices[0].CommittedBytes != 0 {
		t.Fatalf("a failed formation left %d bytes committed on the donor", capacity.Devices[0].CommittedBytes)
	}
	if len(rig.head.Pools()) != 0 {
		t.Fatal("a failed formation left a pool behind")
	}
}

func TestFormRefusesADuplicatePoolID(t *testing.T) {
	rig := newPoolRig(t, 16*mib, 16*mib)

	if _, err := rig.head.Form(context.Background(), "pool-1", rig.modelPath, 128); err != nil {
		t.Fatalf("Form: %v", err)
	}
	t.Cleanup(func() { _ = rig.head.Teardown("pool-1") })

	if _, err := rig.head.Form(context.Background(), "pool-1", rig.modelPath, 128); err == nil {
		t.Fatal("a duplicate pool id should be refused")
	}
}

func TestANodeWithNoServerCommandCannotHead(t *testing.T) {
	rig := newPoolRig(t, 16*mib, 16*mib)
	rig.head.headCommand = nil

	if rig.head.CanHead() {
		t.Fatal("a node with no server command should not report itself able to head")
	}
	if _, err := rig.head.Form(context.Background(), "pool-1", rig.modelPath, 128); err == nil {
		t.Fatal("forming without a server command should be refused")
	}
}

func TestTeardownOfAnUnknownPoolIsAnError(t *testing.T) {
	rig := newPoolRig(t, 16*mib, 16*mib)
	if err := rig.head.Teardown("never-formed"); err == nil {
		t.Fatal("tearing down a pool that does not exist should be an error")
	}
}

// A non-donor's devices are not capacity, so they must not appear in a plan.
func TestOnlyDonatingNodesOfferDevices(t *testing.T) {
	rig := newPoolRig(t, 16*mib, 16*mib)
	devices := rig.head.Devices(context.Background())
	if len(devices) != 2 {
		t.Fatalf("devices = %+v, want one from each node", devices)
	}

	// Turn the head's own donation off and it stops offering.
	rig.head.collector = NewCollector(
		fakeNodeInfo(t, inventory("uuid-head", 16*mib, 0)), &DonorState{}, nil, "uuid-head")
	devices = rig.head.Devices(context.Background())
	for _, d := range devices {
		if d.NodeUUID == "uuid-head" {
			t.Fatal("a node that is not lending should offer no devices")
		}
	}
}

// The head's own devices carry ClassLocal: a measurement of the path to
// ourselves is either absent or meaningless, and letting it read as unknown
// would have the planner refuse the pool its own memory.
func TestTheHeadsOwnDevicesAreLocal(t *testing.T) {
	rig := newPoolRig(t, 16*mib, 16*mib)
	for _, d := range rig.head.Devices(context.Background()) {
		if d.NodeUUID != "uuid-head" {
			continue
		}
		if d.Link != linkq.ClassLocal {
			t.Fatalf("the head's own device has class %v, want local", d.Link)
		}
		if !d.MediaKnown {
			t.Fatal("the head's own media should be known")
		}
	}
}

// Guard the synthetic model against drifting away from what the tests assume.
func TestPoolModelHasTheShapeTheseTestsAssume(t *testing.T) {
	model, err := ReadModel(poolModel(t, t.TempDir()))
	if err != nil {
		t.Fatalf("ReadModel: %v", err)
	}
	if model.Layers != 8 {
		t.Fatalf("Layers = %d, want 8", model.Layers)
	}
	if got := model.NonRepeatingBytes; got != 4*mib {
		t.Fatalf("NonRepeatingBytes = %d, want %d", got, 4*mib)
	}
	if got := model.PerLayerWeightBytes(); got != 2*mib {
		t.Fatalf("PerLayerWeightBytes = %d, want %d", got, 2*mib)
	}
	_ = binary.LittleEndian
}
