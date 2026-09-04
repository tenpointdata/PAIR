// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/linkq"
	"nvpair-shared/poolwire"
	"nvpair-shared/vrampool"
)

// Placeholders a head command template may use, alongside {host} and {port}.
const (
	// PlaceholderModel is the model file's path.
	PlaceholderModel = "{model}"
	// PlaceholderRPC is the comma-separated list of loopback endpoints the
	// donor links listen on — what llama.cpp's --rpc expects.
	PlaceholderRPC = "{rpc}"
	// PlaceholderSplit is the tensor-split proportions, comma-separated.
	PlaceholderSplit = "{split}"
	// PlaceholderContext is the context length the pool was planned for.
	PlaceholderContext = "{context}"
)

// poolTunables are the durations a pool's lifecycle runs on.
//
// They are fields rather than package-level variables because the renewal loop
// and the readiness wait run in their own goroutines: a global that a test
// shortens is a global two goroutines then race over, which the race detector
// finds and which no amount of care at the call site fixes. Set once at
// construction, read freely thereafter.
type poolTunables struct {
	// readyTimeout bounds how long a pool's server has to become ready.
	//
	// Very long on purpose. Forming a pool means pushing the model's weights
	// across the network to every donor — gigabytes, once — and on a wireless
	// link that is tens of minutes. Timing out early would abandon a pool that
	// was working, after paying most of the cost of building it.
	readyTimeout time.Duration
	// readyPoll is how often readiness is retried.
	readyPoll time.Duration
	// stopGrace is how long a pool's server gets to exit on its own.
	stopGrace time.Duration
	// renewInterval is how often the head renews every lease it holds. Well
	// under the minimum TTL, so a single missed renewal — a busy moment, a
	// dropped packet — does not cost the pool.
	renewInterval time.Duration
}

func defaultPoolTunables() poolTunables {
	return poolTunables{
		readyTimeout:  45 * time.Minute,
		readyPoll:     time.Second,
		stopGrace:     10 * time.Second,
		renewInterval: 15 * time.Second,
	}
}

// leaseTTL is what the head asks donors for.
const leaseTTL = 90 * time.Second

// Pool is one formed pool, from the head's point of view.
type Pool struct {
	ID            string             `json:"id"`
	ModelPath     string             `json:"modelPath"`
	ModelName     string             `json:"modelName"`
	ContextTokens int                `json:"contextTokens"`
	Endpoint      string             `json:"endpoint"`
	Members       []PoolMember       `json:"members"`
	Boundaries    int                `json:"boundaries"`
	SlowestLink   linkq.Class        `json:"slowestLink"`
	RequiredBytes uint64             `json:"requiredBytes"`
	Split         string             `json:"tensorSplit"`
	StartedAtMs   int64              `json:"startedAtMs"`
	plan          *vrampool.Plan     `json:"-"`
	donors        []*donorAttachment `json:"-"`
	server        *headProcess       `json:"-"`
	cancel        context.CancelFunc `json:"-"`
}

// PoolMember is one machine's part in a pool, for display.
type PoolMember struct {
	NodeUUID   string      `json:"nodeUuid"`
	NodeName   string      `json:"nodeName"`
	DeviceName string      `json:"deviceName"`
	FirstLayer int         `json:"firstLayer"`
	Layers     int         `json:"layers"`
	Bytes      uint64      `json:"bytes"`
	IsHead     bool        `json:"isHead"`
	Link       linkq.Class `json:"link"`
}

// donorAttachment is one donor's lease plus the loopback link that carries its
// stream.
type donorAttachment struct {
	nodeUUID string
	addr     string
	grant    poolwire.LeaseGrant
	link     *DonorLink
}

// PoolManager is the head side: it plans pools, forms them, keeps their leases
// alive, and tears them down.
//
// It is deliberately separate from LeaseStore even though both live in this
// binary. They are the two ends of the same protocol and a node plays both roles
// — but a node heads the pools its own users ask for and donates to pools other
// people head, and conflating the two states is how a donor ends up planning
// against memory it has already lent.
type PoolManager struct {
	mesh      *clustertrust.Mesh
	peers     *PeerCollector
	collector *Collector
	leases    *LeaseClient
	// headCommand is the argv template that runs a pool's server. Empty means
	// this node cannot head a pool, which is a different thing from being unable
	// to donate to one.
	headCommand []string
	policy      vrampool.Policy
	tunables    poolTunables

	mu    sync.Mutex
	pools map[string]*Pool
}

func NewPoolManager(mesh *clustertrust.Mesh, peers *PeerCollector, collector *Collector, headCommand []string, policy vrampool.Policy) *PoolManager {
	return &PoolManager{
		mesh:        mesh,
		peers:       peers,
		collector:   collector,
		leases:      NewLeaseClient(mesh),
		headCommand: headCommand,
		policy:      policy,
		tunables:    defaultPoolTunables(),
		pools:       make(map[string]*Pool),
	}
}

// CanHead reports whether this node is able to run a pool's server at all.
func (m *PoolManager) CanHead() bool { return len(m.headCommand) > 0 }

// Devices assembles the planner's view of the cluster from this node's capacity
// and the last collected peer capacity.
//
// The head's own devices come first and carry ClassLocal, because a measurement
// of the path to ourselves is either absent or meaningless. Peers carry whatever
// the capacity sweep observed, which is what makes the planner's donor gate
// operate on evidence rather than on an assumption that every peer is equal.
func (m *PoolManager) Devices(ctx context.Context) []vrampool.Device {
	local := m.collector.Local(ctx)
	out := devicesFor(local, linkq.Path{Class: linkq.ClassLocal, Stability: 1, MediaKnown: true})

	for uuid, capacity := range m.peers.Last() {
		path, ok := m.peers.Path(uuid)
		if !ok {
			// Nothing measured: leave the class unknown rather than optimistic.
			// The default policy refuses an unmeasured donor, which is the point.
			path = linkq.Path{}
		}
		out = append(out, devicesFor(capacity, path)...)
	}
	// Stable order so a plan is reproducible between calls on an unchanged
	// cluster; the planner's own ordering then decides who is filled first.
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].NodeUUID != out[j].NodeUUID {
			return out[i].NodeUUID < out[j].NodeUUID
		}
		return out[i].Index < out[j].Index
	})
	return out
}

// devicesFor renders one node's capacity as planner devices.
func devicesFor(capacity poolwire.NodeCapacity, path linkq.Path) []vrampool.Device {
	if !capacity.DonorEnabled {
		return nil
	}
	out := make([]vrampool.Device, 0, len(capacity.Devices))
	for _, d := range capacity.Devices {
		out = append(out, vrampool.Device{
			NodeUUID:   capacity.NodeUUID,
			NodeName:   capacity.NodeName,
			Index:      d.Index,
			Name:       d.Name,
			TotalBytes: d.TotalBytes,
			FreeBytes:  d.FreeBytes(),
			Link:       path.Class,
			Stability:  path.Stability,
			MediaKnown: path.MediaKnown,
		})
	}
	return out
}

// PlanResult is a dry run: what a pool for this model would look like, or why it
// cannot be formed.
type PlanResult struct {
	Model          vrampool.Model       `json:"model"`
	ContextTokens  int                  `json:"contextTokens"`
	Fits           bool                 `json:"fits"`
	FitsOnOneNode  bool                 `json:"fitsOnOneNode"`
	Members        []PoolMember         `json:"members,omitempty"`
	Rejected       []vrampool.Rejection `json:"rejected,omitempty"`
	ShortfallBytes uint64               `json:"shortfallBytes,omitempty"`
	Boundaries     int                  `json:"boundaries"`
	SlowestLink    linkq.Class          `json:"slowestLink"`
	TensorSplit    string               `json:"tensorSplit,omitempty"`
}

// Plan answers what a pool for this model would look like without forming one.
//
// It reports FitsOnOneNode separately, and that is the field a caller should
// look at first: a model that fits on a single node should be ROUTED there, not
// pooled. Routing is faster, survives that node going away, and is what PAIR
// already does well. Pooling is for the case where the alternative is not
// running at all.
func (m *PoolManager) Plan(ctx context.Context, modelPath string, contextTokens int) (PlanResult, error) {
	model, err := ReadModel(modelPath)
	if err != nil {
		return PlanResult{}, err
	}
	devices := m.Devices(ctx)

	result := PlanResult{Model: model, ContextTokens: contextTokens}
	if _, ok := vrampool.FitsOnOne(model, contextTokens, devices, m.policy); ok {
		result.FitsOnOneNode = true
	}

	outcome, err := vrampool.Build(model, contextTokens, devices, m.collector.NodeUUID(), m.policy)
	if err != nil {
		return result, err
	}
	result.Rejected = outcome.Rejected
	result.ShortfallBytes = outcome.ShortfallBytes
	if !outcome.Fits() {
		return result, nil
	}

	result.Fits = true
	result.Boundaries = outcome.Plan.Boundaries
	result.SlowestLink = outcome.Plan.SlowestLink
	result.TensorSplit = outcome.Plan.TensorSplitArg()
	result.Members = membersOf(outcome.Plan)
	return result, nil
}

// membersOf renders a plan's assignments for display.
func membersOf(plan *vrampool.Plan) []PoolMember {
	out := make([]PoolMember, 0, len(plan.Assignments))
	for _, a := range plan.Assignments {
		out = append(out, PoolMember{
			NodeUUID:   a.Device.NodeUUID,
			NodeName:   a.Device.NodeName,
			DeviceName: a.Device.Name,
			FirstLayer: a.FirstLayer,
			Layers:     a.Layers,
			Bytes:      a.TotalBytes,
			IsHead:     a.IsHead,
			Link:       a.Device.Link,
		})
	}
	return out
}

// Form plans a pool and brings it up: leases every donor, opens their tunnels,
// and starts the server.
//
// Failure at any step tears down everything already brought up. A half-formed
// pool is worse than none: donors hold memory for a head that is not coming, and
// the memory only returns when their leases lapse.
func (m *PoolManager) Form(ctx context.Context, poolID, modelPath string, contextTokens int) (*Pool, error) {
	if !m.CanHead() {
		return nil, errors.New("this node has no pool server command configured")
	}
	if poolID == "" {
		return nil, errors.New("no pool id")
	}

	m.mu.Lock()
	if _, exists := m.pools[poolID]; exists {
		m.mu.Unlock()
		return nil, fmt.Errorf("pool %s already exists", poolID)
	}
	m.mu.Unlock()

	model, err := ReadModel(modelPath)
	if err != nil {
		return nil, err
	}
	// Devices first, as its own statement: reading capacity is also what teaches
	// this node its own identity from node-info, and passing both as arguments
	// would leave that dependency resting on argument evaluation order.
	devices := m.Devices(ctx)
	outcome, err := vrampool.Build(model, contextTokens, devices, m.collector.NodeUUID(), m.policy)
	if err != nil {
		return nil, err
	}
	if !outcome.Fits() {
		return nil, &NotEnoughMemoryError{Outcome: outcome, Model: model, ContextTokens: contextTokens}
	}
	plan := outcome.Plan

	poolCtx, cancel := context.WithCancel(context.Background())
	pool := &Pool{
		ID:            poolID,
		ModelPath:     modelPath,
		ModelName:     model.Name,
		ContextTokens: contextTokens,
		Members:       membersOf(plan),
		Boundaries:    plan.Boundaries,
		SlowestLink:   plan.SlowestLink,
		RequiredBytes: plan.RequiredBytes,
		Split:         plan.TensorSplitArg(),
		StartedAtMs:   time.Now().UnixMilli(),
		plan:          plan,
		cancel:        cancel,
	}

	if err := m.attachDonors(poolCtx, pool, plan, model, contextTokens); err != nil {
		cancel()
		m.detach(pool)
		return nil, err
	}

	server, err := m.startServer(poolCtx, pool)
	if err != nil {
		cancel()
		m.detach(pool)
		return nil, err
	}
	pool.server = server
	pool.Endpoint = server.endpoint

	m.mu.Lock()
	m.pools[poolID] = pool
	m.mu.Unlock()

	go m.renewLoop(poolCtx, pool)

	slog.Info("pool formed", "pool", poolID, "model", model.Name,
		"nodes", len(plan.Nodes()), "boundaries", plan.Boundaries,
		"slowestLink", plan.SlowestLink.String(), "endpoint", server.endpoint)
	return pool, nil
}

// NotEnoughMemoryError reports a model that will not fit, with everything needed
// to explain why.
type NotEnoughMemoryError struct {
	Outcome       vrampool.Outcome
	Model         vrampool.Model
	ContextTokens int
}

func (e *NotEnoughMemoryError) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s does not fit: %d of %d layers had nowhere to go, about %.1f GiB short",
		e.Model.Name, e.Outcome.UnplacedLayers, e.Model.Layers, float64(e.Outcome.ShortfallBytes)/(1<<30))
	for _, r := range e.Outcome.Rejected {
		fmt.Fprintf(&b, "; %s excluded (%s)", r.Device.NodeName, r.Reason)
	}
	return b.String()
}

// attachDonors leases every non-head member and opens its tunnel.
func (m *PoolManager) attachDonors(ctx context.Context, pool *Pool, plan *vrampool.Plan, model vrampool.Model, contextTokens int) error {
	// One lease per donor NODE, not per device: a machine with two cards in the
	// pool is one lease covering both, because the memory and the backend
	// process are the machine's, not the card's.
	byNode := map[string][]vrampool.Assignment{}
	var order []string
	for _, a := range plan.Assignments {
		if a.IsHead {
			continue
		}
		if _, seen := byNode[a.Device.NodeUUID]; !seen {
			order = append(order, a.Device.NodeUUID)
		}
		byNode[a.Device.NodeUUID] = append(byNode[a.Device.NodeUUID], a)
	}

	for _, nodeUUID := range order {
		assignments := byNode[nodeUUID]
		addr, ok := m.peers.Address(nodeUUID)
		if !ok {
			return fmt.Errorf("no address for donor %s", assignments[0].Device.NodeName)
		}

		devices := make([]int, 0, len(assignments))
		var bytes uint64
		for _, a := range assignments {
			devices = append(devices, a.Device.Index)
			bytes += a.TotalBytes
		}

		grant, err := m.leases.Request(ctx, nodeUUID, addr, poolwire.LeaseRequest{
			PoolID:        pool.ID,
			ModelName:     model.Name,
			DeviceIndexes: devices,
			Bytes:         bytes,
			TTLMs:         leaseTTL.Milliseconds(),
		})
		if err != nil {
			return fmt.Errorf("lease %s: %w", assignments[0].Device.NodeName, err)
		}

		link := NewDonorLink(m.mesh, nodeUUID, addr, grant.LeaseID)
		if err := link.Start(ctx); err != nil {
			// Release immediately rather than letting it lapse: the donor has a
			// backend process running for a pool that is not going to exist.
			_ = m.leases.Release(ctx, nodeUUID, addr, grant.LeaseID)
			return fmt.Errorf("open tunnel to %s: %w", assignments[0].Device.NodeName, err)
		}

		pool.donors = append(pool.donors, &donorAttachment{
			nodeUUID: nodeUUID, addr: addr, grant: grant, link: link,
		})
	}
	return nil
}

// detach tears down every donor a pool holds, in the order that leaves nothing
// running: the links first, then the leases.
func (m *PoolManager) detach(pool *Pool) {
	ctx, cancel := context.WithTimeout(context.Background(), leaseCallTimeout)
	defer cancel()
	for _, d := range pool.donors {
		if d.link != nil {
			_ = d.link.Close()
		}
		if err := m.leases.Release(ctx, d.nodeUUID, d.addr, d.grant.LeaseID); err != nil {
			slog.Warn("pool: releasing a donor lease failed; it will lapse on its own",
				"pool", pool.ID, "donor", d.nodeUUID, "err", err)
		}
	}
	pool.donors = nil
}

// renewLoop keeps every lease in a pool alive for as long as the pool is.
//
// A donor whose renewal stops answering is not merely a stale entry: its lease
// will lapse, its backend will be torn down, and the pool's next token will fail
// with it. There is no re-sharding in llama.cpp, so there is nothing to do about
// it except say so loudly.
func (m *PoolManager) renewLoop(ctx context.Context, pool *Pool) {
	ticker := time.NewTicker(m.tunables.renewInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			for _, d := range pool.donors {
				renewCtx, cancel := context.WithTimeout(ctx, leaseCallTimeout)
				_, err := m.leases.Renew(renewCtx, d.nodeUUID, d.addr, d.grant.LeaseID, leaseTTL)
				cancel()
				if err != nil {
					slog.Error("pool: donor lease renewal failed; the pool will fail when it lapses",
						"pool", pool.ID, "donor", d.nodeUUID, "err", err)
				}
			}
		}
	}
}

// Teardown ends a pool.
//
// The server first, then the donors. The other order leaves a running server
// holding streams to backends that are shutting down, which surfaces as a
// truncated tensor transfer rather than as the orderly stop it was asked for.
func (m *PoolManager) Teardown(poolID string) error {
	m.mu.Lock()
	pool, ok := m.pools[poolID]
	delete(m.pools, poolID)
	m.mu.Unlock()
	if !ok {
		return fmt.Errorf("no pool %s", poolID)
	}

	if pool.cancel != nil {
		pool.cancel()
	}
	if pool.server != nil {
		pool.server.stop(m.tunables.stopGrace)
	}
	m.detach(pool)
	slog.Info("pool torn down", "pool", poolID)
	return nil
}

// TeardownAll ends every pool, for shutdown.
func (m *PoolManager) TeardownAll() {
	m.mu.Lock()
	ids := make([]string, 0, len(m.pools))
	for id := range m.pools {
		ids = append(ids, id)
	}
	m.mu.Unlock()
	for _, id := range ids {
		_ = m.Teardown(id)
	}
}

// Pools returns the formed pools, for status.
func (m *PoolManager) Pools() []Pool {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]Pool, 0, len(m.pools))
	for _, p := range m.pools {
		out = append(out, *p)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// headProcess is a pool's server.
type headProcess struct {
	cmd      *exec.Cmd
	endpoint string
	done     chan struct{}
}

// startServer runs the pool's server against the donor links.
func (m *PoolManager) startServer(ctx context.Context, pool *Pool) (*headProcess, error) {
	port, err := freeLoopbackPort()
	if err != nil {
		return nil, fmt.Errorf("reserve server port: %w", err)
	}

	rpcEndpoints := make([]string, 0, len(pool.donors))
	for _, d := range pool.donors {
		rpcEndpoints = append(rpcEndpoints, d.link.Addr())
	}

	argv := make([]string, len(m.headCommand))
	for i, arg := range m.headCommand {
		arg = strings.ReplaceAll(arg, PlaceholderHost, donorHost)
		arg = strings.ReplaceAll(arg, PlaceholderPort, strconv.Itoa(port))
		arg = strings.ReplaceAll(arg, PlaceholderModel, pool.ModelPath)
		arg = strings.ReplaceAll(arg, PlaceholderRPC, strings.Join(rpcEndpoints, ","))
		arg = strings.ReplaceAll(arg, PlaceholderSplit, pool.Split)
		arg = strings.ReplaceAll(arg, PlaceholderContext, strconv.Itoa(pool.ContextTokens))
		argv[i] = arg
	}

	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Env = os.Environ()
	// Same rule as every engine PAIR starts: never inherit stdout, which carries
	// this process's JSON-RPC frames.
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start pool server: %w", err)
	}

	proc := &headProcess{
		cmd:      cmd,
		endpoint: "http://" + donorHost + ":" + strconv.Itoa(port),
		done:     make(chan struct{}),
	}
	go func() {
		_ = cmd.Wait()
		close(proc.done)
	}()

	if err := waitForServing(ctx, proc.endpoint, m.tunables); err != nil {
		proc.stop(m.tunables.stopGrace)
		return nil, fmt.Errorf("pool server never became ready: %w", err)
	}
	return proc, nil
}

// stop ends a pool's server.
func (p *headProcess) stop(grace time.Duration) {
	if p.cmd.Process == nil {
		return
	}
	_ = p.cmd.Process.Signal(os.Interrupt)
	select {
	case <-p.done:
		return
	case <-time.After(grace):
	}
	_ = p.cmd.Process.Kill()
	<-p.done
}

// waitForServing blocks until endpoint answers an HTTP request with any status
// other than "still loading".
//
// A TCP handshake is not enough here, unlike on the donor side. llama.cpp's
// server binds its port as soon as it starts and answers 503 for the whole time
// the model is loading — which across a pool is the slowest part by far. Treating
// the bind as readiness would hand the endpoint to a proxy minutes before it can
// serve anything.
func waitForServing(ctx context.Context, endpoint string, t poolTunables) error {
	client := &http.Client{Timeout: t.readyPoll * 5}
	deadline := time.Now().Add(t.readyTimeout)
	var lastStatus string

	for {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint+"/health", nil)
		if err != nil {
			return err
		}
		resp, err := client.Do(req)
		if err == nil {
			status := resp.StatusCode
			closeBody(resp)
			if status != http.StatusServiceUnavailable {
				return nil
			}
			lastStatus = resp.Status
		}
		if time.Now().After(deadline) {
			if lastStatus != "" {
				return fmt.Errorf("timed out after %s, last answer %s", t.readyTimeout, lastStatus)
			}
			return fmt.Errorf("timed out after %s", t.readyTimeout)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(t.readyPoll):
		}
	}
}
