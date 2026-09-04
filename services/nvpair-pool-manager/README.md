<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvpair-pool-manager

The capacity plane for distributed inference: what this node is willing to
contribute to a pool, what it currently could contribute, and the cluster-scoped
surface peers read that from.

Refer to [GPU pooling](../../docs/gpu-pooling.mdx) for the design this
implements and, more importantly, for what pooling buys and costs. The short
version: it buys **capacity**, it costs **latency**, and it is for the case where
the alternative is not running the model at all. A model that fits on one node
should be routed there, which is what PAIR already does well.

Every node runs this service and plays both roles. As a **donor** it lends VRAM:
it reports capacity, grants leases, runs a ggml RPC backend per lease bound to
loopback, and carries that backend's stream to the head over cluster mTLS. As a
**head** it forms pools: it plans across the cluster, leases the donors it needs,
opens their tunnels, starts a server against them, and keeps the leases alive for
as long as the pool lives.

## Why this is its own service

Two facts about pool capacity decide the shape of this component, and both push
the same way.

**It is not hardware inventory.** `nvpair-node-info` reports what the driver
says, in plaintext, to anything on the subnet — a documented trade for hostname,
GPU model, and utilization. "This node has 40 GiB committed to a pool headed by
that node" is not that. It describes the shape of a user's cluster and what their
machines have promised each other, so it is served over cluster mTLS from this
port instead, readable only by pinned members.

**Lending memory is a decision, not a measurement.** Donation is off until the
machine's owner turns it on, and the reservation they hold back is theirs to set.
Neither is derivable from anything, so both are stored here.

## Communication

Bidirectional newline-delimited JSON-RPC 2.0 with the parent broker, over
stdin/stdout by default or a Unix socket / Windows named pipe via `--ipc`. Peers
are reached over HTTPS on the cluster port; peers never touch the JSON-RPC
channel.

`nvpair-ui-broker` supervises this worker like any other, passes it the cluster
directory and the llama.cpp command templates, registers its port as the `pl`
service so peers can find it, and relays every method below. It also pushes the
peer set down: discovery is the scanner's job and the broker already holds the
consolidated directory, so a second mDNS browser here would compete for the same
multicast socket to learn what the parent already knows.

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--ipc <path>` | _(stdio)_ | IPC endpoint: Unix socket or Windows named pipe |
| `--port <n>` | `14324` | Cluster-mTLS port. `0` binds an OS-chosen port |
| `--cluster-dir <path>` | _(none)_ | Cluster config dir. Peers are authenticated against the pins it holds; without it this node serves nothing |
| `--node-info-url <url>` | `http://127.0.0.1:14318/v1/node-info` | Local hardware inventory |
| `--donor-settings <path>` | _(per-user data dir)_ | Path to `pool-donor.json` |
| `--donor-command <template>` | _(none)_ | Command that runs a ggml RPC backend for one lease. Without it this node cannot donate, and its lease endpoints answer `503` |
| `--head-command <template>` | _(none)_ | Command that runs a pool's server. Without it this node cannot head a pool |
| `--allow-wifi-donors` | `false` | Let a pool use a donor reached over a wireless link |
| `--pool-headroom <fraction>` | _(built-in)_ | Fraction of a device's free VRAM a pool may claim |
| `--log-level <level>` | `info` | `error` / `warn` / `info` / `debug`; falls back to `$NVPAIR_LOG_LEVEL`. Also changeable at runtime via `log/set-level` |
| `--version` | | Print version and exit |

## HTTP surface

Four endpoints, all on the cluster port, all **cluster mTLS in every state**.

### `GET /v1/pool/capacity`

Returns this node's poolable capacity. The caller must present a certificate
this node currently pins; anything else gets `403`, and a node that belongs to no
cluster presents no leaf at all and fails the handshake.

```json
{
  "nodeUuid": "…",
  "nodeName": "studio",
  "donorEnabled": true,
  "devices": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 5090",
      "totalBytes": 34359738368,
      "usedBytes": 4294967296,
      "reservedBytes": 2147483648,
      "committedBytes": 0
    }
  ],
  "observedAtMs": 1772668800000,
  "telemetryValid": true
}
```

The four byte counts are reported separately rather than pre-subtracted into one
free figure, because they answer different questions: `usedBytes` is what is
happening, `reservedBytes` is what the owner declined to lend, and
`committedBytes` is what an existing lease already took. Collapsing them makes
"why is my 24 GiB card offering 4 GiB" unanswerable.

`committedBytes` is tracked apart from `usedBytes` because a lease is a promise
made before the memory is touched. Between granting one and the model finishing
its load, the driver still reports the memory free, and a second pool planned
against it would overcommit the device.

There is **no plaintext personality** on this port. Everything it carries is
cluster state, and the tunnel below fronts a protocol whose only protection is
that nothing unpinned can open the socket.

### `POST /v1/pool/leases`

Asks this node to hold memory for one pool. Answers a `LeaseGrant`, or `409`
with a reason a user can act on — "device 0 has 6 GiB free, needs 14" is the
whole point of asking. `403` if this node is not lending, `503` if it has no
backend command configured.

One pool at a time. `rpc-server` has no concept of two callers, and the memory a
second pool plans against is memory the first is about to allocate; a donor that
accepted both would fail both. A repeat request from the same holder for the same
pool is a **renewal**, not a second lease, so a head that retried through a
dropped response does not double-charge the device it is already using.

The backend process is started *before* the grant is returned. A grant the head
believes in but whose process never came up is the worst failure available here:
the head plans a pool around memory that will never answer and finds out only
once the model is loading everywhere else.

### `POST /v1/pool/leases/{id}` and `DELETE /v1/pool/leases/{id}`

Renew and release. A lease lapses without renewal — that is how a head that dies
returns a donor's memory with no cleanup message from a process that is gone.
Both refuse a lease held by someone else, and refuse it as *not found* rather
than *forbidden*: the distinction would confirm the existence of a lease the
caller has no business knowing about.

### `POST /v1/pool/rpc/{id}`

An HTTP/1.1 upgrade to `nvpair-ggml-rpc` that becomes the tensor stream for a
lease.

This is the most security-sensitive handler in PAIR. On the far side of the
splice is `rpc-server`: no authentication, and it deserializes tensor and buffer
descriptors straight into a compute backend, which is why upstream llama.cpp is
unambiguous that it must never face an untrusted network. Four things hold before
a single byte is spliced — the connection is cluster mTLS, the caller presents a
certificate this node currently pins, the lease exists, and **the caller is the
node the lease was issued to**. That last one is not redundant: passing the
cluster gate authorizes a member to *ask* for a lease, not to use one another
member holds.

## The donor backend

`--donor-command` is an argv template, expanded and executed directly. There is
no shell, so a semicolon or a backtick in it is an argument, not a command.

| Placeholder | Becomes |
|---|---|
| `{host}` | `127.0.0.1`, always |
| `{port}` | The port chosen for this lease |
| `{devices}` | The leased device indexes, comma-separated |

`CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` are also set to the leased
devices, because backends differ in which they honor and setting both is
harmless.

```
--donor-command "/opt/llama.cpp/bin/rpc-server -H {host} -p {port}"
```

**The backend must bind `{host}` and nothing else, and this is verified rather
than trusted.** After it starts accepting, PAIR tries to reach the same port from
each of this machine's non-loopback addresses. If any of them answers, the
backend bound a wildcard address — putting an unauthenticated, memory-unsafe
tensor protocol on every interface the machine has — and it is killed and the
lease refused. The check exists because that mistake is easy to make, works
perfectly, and is therefore invisible.

Each lease gets its own process on its own OS-chosen port, so two pools on one
machine do not collide.

## The pool server

`--head-command` is the same kind of template, with more placeholders:

| Placeholder | Becomes |
|---|---|
| `{host}`, `{port}` | The loopback address the pool will serve on |
| `{model}` | The model file's path |
| `{rpc}` | The loopback donor-link endpoints, comma-separated |
| `{split}` | The planned tensor-split proportions |
| `{context}` | The context length the pool was planned for |

```
--head-command "/opt/llama.cpp/bin/llama-server -m {model} --host {host} --port {port} --rpc {rpc} --tensor-split {split} -c {context} -ngl 999"
```

`{rpc}` is the whole trick. llama.cpp cannot speak TLS — `--rpc host:port` opens a
plain TCP connection and sends the ggml protocol on it — so it is given loopback
addresses that are not on the network, and this service is what crosses the
network on its behalf.

## What decides which donors a pool uses

Planning is `nvpair-shared/vrampool`, and its inputs come from two places. Free
VRAM comes from each node's own capacity report. Link quality comes from the
capacity sweep itself: a sweep is a timed request that was going to happen
anyway, so its round-trip time and its success or failure are free measurements,
and `nvpair-shared/linkq` turns them into a class.

Link quality is a **gate**, not a preference, and the reason is not performance.
llama.cpp has no re-sharding and no failover, so a donor that drops takes the
pool and the in-flight request with it. A wide-area donor is refused outright; a
wireless one needs `--allow-wifi-donors`; an unmeasured link is refused because
admitting a donor on no evidence is how a pool acquires the member that kills it.

## JSON-RPC methods (broker → manager)

### `pool/status`

This node's donation policy, its own capacity, its outstanding leases, whether it
has a backend command at all, and the last collected view of its peers'. Does not
re-ask peers.

`donorReady` is reported separately from `donor.enabled` on purpose: a node that
has enabled donation but cannot run a backend is a different problem from one
that has declined, and collapsing them hides a misconfiguration behind a switch
that looks correct.

### `pool/capacity`

The same shape, after re-asking every known peer.

Both answer with `{donor, donorReady, local, peers, leases, clusterFreeBytes}`. One shape for both,
because the only difference is whether peers were refreshed, and a caller that
has to branch on which call it made to read the answer is a caller that will get
it wrong.

### `pool/donor/set`

```json
{"jsonrpc":"2.0","id":1,"method":"pool/donor/set","params":{"enabled":true,"reservedBytes":2147483648}}
```

Both params are optional and independent: sending one leaves the other alone, so
turning donation off does not silently clear a reservation set months ago. A
reservation larger than 1 TiB is refused as a unit error — bytes typed where
gigabytes were meant would disable donation on every device while the switch
still read as on.

Persisted atomically before the response is sent, then announced as
`pool/updated`. A no-op re-assert answers but announces nothing.

### `pool/peers/set`

```json
{"jsonrpc":"2.0","id":2,"method":"pool/peers/set","params":{"peers":[{"nodeUuid":"…","addresses":["10.0.0.9:14324"]}]}}
```

The peer set is pushed down rather than discovered here. Discovery is the
scanner's job and the broker already holds the consolidated directory; a second
mDNS browser in this process would compete for the same multicast socket to learn
something the parent already knows. Replacing rather than merging is deliberate —
a member that has left must stop being asked, and its last capacity reading is
dropped with it, because a remembered one would let a planner build a pool around
a machine that is gone.

Addresses are a ranked list, walked by `nvpair-shared/reach`, because a
multi-homed peer has no single address every node can reach.

### `pool/plan`

```json
{"jsonrpc":"2.0","id":6,"method":"pool/plan","params":{"modelPath":"/models/big.gguf","contextTokens":8192}}
```

Answers what a pool for that model would look like — which machines, how many
layers each, how many machine boundaries, the worst link involved — without
forming anything. It reads the model's GGUF **header only**, never a tensor, so
it answers before gigabytes move.

`fitsOnOneNode` is the field to read first. When it is true, route the request
instead: one node can hold the model, and routing is faster, survives that node
going away, and needs none of this. When the model fits nowhere, the answer
carries how far short it fell and a reason per excluded device.

### `pool/form` and `pool/teardown`

Bring a pool up and end it. Forming leases every donor, opens its tunnel, starts
the server, and waits for it to actually serve — not merely to bind, because
llama.cpp's server answers `503` for the whole time the model is loading, which
across a pool is by far the slowest part.

Failure at any step tears down everything already brought up. A half-formed pool
is worse than none: donors hold memory for a head that is not coming, and it only
returns when their leases lapse.

Teardown stops the server first and the donors second. The other order leaves a
running server holding streams to backends that are shutting down, which surfaces
as a truncated transfer rather than the orderly stop it was asked for.

### `shutdown`, `log/set-level`

As every other worker.

## Notifications (manager → broker)

| Method | When |
|---|---|
| `ready` | Once at startup, with the version |
| `pool/updated` | After donation policy changes, carrying the same shape as `pool/status` |

## Donation policy on disk

`pool-donor.json` in the per-user PAIR data directory, owner-only, written
atomically.

It is its own file rather than a key in `nvpair-node-settings` because donation
is the one setting in PAIR that hands another machine the ability to allocate
memory in a process on this one. That is worth being able to answer by reading a
single file, and worth having survive a settings store that is reset or migrated.

**Every path that cannot read a clear yes answers no.** A missing file, an empty
one, and a corrupt one all read as donation off. A corrupt file does not fail the
process, because the safe reading and the conservative one are the same here.

## Testing

```bash
go test ./...
```

The cluster-mTLS surface is exercised end to end against real certificates via
`nvpair-shared/clustertrusttest`: a pinned peer is served, an unpinned caller that
completes a handshake gets `403`, a plaintext request is never seen by a handler,
an unclustered node refuses every handshake, and removing a peer's pin takes
effect on the next request rather than the next restart.

The tunnel is tested through the real path rather than in pieces: a head opens a
`DonorLink`, three megabytes of random bytes cross a loopback listener, a cluster
mTLS connection, and a loopback backend, and come back identical. A cluster member
that holds no lease gets `404`, an unpinned caller gets `403`, a request without
the protocol token gets `400`, and releasing a lease severs a stream that is
already carrying traffic.

The backend runner re-executes the test binary as its child, the way `os/exec`'s
own tests do, so process startup, readiness, termination, and the wildcard-bind
refusal all run against a real process without needing a llama.cpp build. The
wildcard test skips itself on a host with no non-loopback address, since it would
have nothing to probe from.

Formation is tested as a real two-machine pool. The stand-in pool server reaches
every `--rpc` endpoint it was given and only then reports itself healthy, so a
passing test means the whole path worked: placeholder expansion, the loopback
link, cluster mTLS, and the donor's own backend. The model is a synthetic GGUF
built in the test — a real one is gigabytes, and a truncated one would not
exercise the tensor table the size estimate depends on.
