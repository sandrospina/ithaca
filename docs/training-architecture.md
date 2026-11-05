# Training Architecture — Shared GPU Servers with Local Fallback

Decisions:
- 2026-11-02: training can run on the developer's GPU servers, with fallback to local hardware.
- 2026-11-02: **multiple players** use these servers. Players pick a server to train on.
- 2026-11-02: access is **invite-only**: friends, students or a class. Public access is out of scope, but the design should not prevent it later.

## 1. Hardware inventory

| Machine | CPU arch | GPU | Memory | Preferred backend | Fallback backend |
|---|---|---|---|---|---|
| DGX Spark | ARM64 (Grace, 20 cores) | Blackwell GB10 | 128 GB unified | CUDA | wgpu / Vulkan |
| RTX 2080 PC | x86-64 | Turing, sm_75 | 8 GB VRAM | CUDA (fp32/fp16; no bf16) | wgpu / Vulkan |
| Radeon RX 9070 XT PC (Windows 11) | x86-64 | RDNA4, gfx1201 | 16 GB VRAM | ROCm/HIP in WSL2 (to be verified) | wgpu native on Windows (Vulkan/DX12) |
| Player's browser | any | any | — | WebGPU | WASM CPU in Web Workers |

The DGX Spark and the RTX 2080 PC are assumed to run Linux. The RX 9070 XT PC runs Windows 11 (confirmed 2026-11-02).

### RX 9070 XT on Windows

- AMD officially supports ROCm in WSL2 for the RX 9000 series, through the open-source ROCDXG library. This needs ROCm 7.2.1 or newer with the Adrenalin 26.2.2 driver or newer (status checked 2026-11-02).
- Unverified: whether Burn's ROCm backend works on top of ROCDXG in WSL2. No source documents it either way.
- Guaranteed baseline: run the worker **natively on Windows** with Burn's wgpu backend. It needs no WSL and no ROCm.
- The hardware spike tests both paths on this machine. The worker uses whichever is faster and stable.
- The worker connects outbound to the coordinator, so WSL2's awkward inbound networking is never involved.
- Operational notes for a Windows desktop:
  - Run the worker as a service or a Task Scheduler task that starts at boot.
  - Configure WSL2 so its VM is not shut down while idle.
  - Stop Windows Update from restarting the machine during active hours.
  - Add a "drain" switch, so the owner can stop new jobs before gaming or other heavy use. Running jobs finish or checkpoint, then the worker goes offline.

## 2. Deep-learning framework: Burn

| Requirement | Burn | Candle |
|---|---|---|
| NVIDIA CUDA (Spark, 2080) | yes | yes |
| AMD ROCm (9070 XT) | yes | no |
| Vulkan fallback on any GPU | yes (wgpu) | no |
| WebGPU in the browser | yes (wgpu) | no |
| One model definition across all of the above | yes, the backend is chosen at run time by the `Device` | no |

Model code is written once and runs on whichever `Device` it is given. Since Burn 0.22 there is no backend type
parameter: `Device::cuda(0)`, `Device::rocm(0)`, `Device::vulkan(..)`, `Device::webgpu(..)` or `Device::flex()`
(pure-Rust CPU), each behind its Cargo feature, and `device.autodiff()` for training.

**Risk to spike early:** Burn's CUDA backend on the DGX Spark (ARM64 host + Blackwell + CUDA 13). Run a small Burn training example on all three servers and in the browser before Milestone 8.

### Hardware spike (started 2026-11-04)

The spike program is `spikes/burn/`; its README has the command for each machine. It trains the same small network
on each backend, fitted to the Q-table the game's tabular Q-learning learns on Level 1, and checks a matmul against
an f64 reference, the network's greedy policy, bit-determinism, and a benchmark loss recomputed on the host.

Burn 0.22.1, CubeCL 0.11.3.

| Machine | Backend | Result | Deterministic | Throughput (256→512→512→4, batch 4096) | Notes |
|---|---|---|---|---|---|
| Developer Mac, Apple M5 | Metal (native wgpu) | PASS | yes | 1.1 TFLOP/s; 1.3 with autotune | autotune costs 10–20 s on first use |
| Developer Mac, Apple M5 | Flex (CPU) | PASS | yes | 0.18–0.29 TFLOP/s at batch 1024 | the CPU fallback |
| Chrome (headless), Apple M5 | WebGPU, fusion, **no autotune** | PASS (3 of 3 runs) | yes | 30–52 GFLOP/s | fit 9.8 s warm |
| Chrome (headless), Apple M5 | WebGPU with autotune | **wrong results in 4 of 5 runs** | no | — | see below |
| DGX Spark | CUDA | not run yet | | | |
| DGX Spark | Vulkan | not run yet | | | fallback |
| RTX 2080 PC | CUDA | not run yet | | | |
| RTX 2080 PC | Vulkan | not run yet | | | fallback |
| RX 9070 XT, Windows | Vulkan / DX12 (native wgpu) | not run yet | | | guaranteed baseline |
| RX 9070 XT, WSL2 | ROCm | not run yet | | | |

Findings so far:

- **Browser training must turn kernel autotuning off.** With it on (Burn's default), one CubeCL matmul variant fails
  WGSL validation in Chrome, does no work, times as fastest and is sometimes chosen. Results are then silently wrong:
  a negative mean-squared loss, NaN, a network that disagrees with its target. Build the browser with
  `default-features = false` and `fusion`, without `autotune`. Fusion is safe and makes the small fit 3× faster.
- WebGPU in the browser is about 20–40× slower than native Metal on the same GPU for this network. Small networks
  (Milestone 8's DQN levels) are fine locally; anything larger belongs on the servers, as planned.
- The CUDA and ROCm backends compile kernels through an LLVM that Burn bundles; no system LLVM is needed on Linux.
  (On macOS the bundled LLVM needs Homebrew's zstd, which is why these two builds were not compile-checked here.)
- Every check that can fail does so loudly: a backend counts as working only with `PASS` from the spike, not just
  because it starts.

Exit criterion still open: a `PASS` line for each server, pasted into the table above.

## 3. Topology: coordinator + workers

With many players, the GPU machines must not be exposed directly to the internet. A central **coordinator** is the only public service. GPU machines run **workers** that connect *outbound* to the coordinator and pull jobs.

```text
 Players' browsers (Leptos/WASM)
        │  HTTPS / WSS  (accounts, jobs, live metrics, policy library)
        ▼
 ┌───────────────────────── calypso-coordinator (public) ──────────────────────────┐
 │ static web app · auth & accounts · server directory · per-device job queues     │
 │ quotas & fair scheduling · policy/experiment storage · metrics relay            │
 └─────────────────────────────────────────────────────────────────────────────────┘
        ▲                     ▲                       ▲
        │ outbound WSS        │ outbound WSS          │ outbound WSS   (worker tokens)
 calypso-worker          calypso-worker          calypso-worker
 dgx-spark [CUDA]        rtx-2080 [CUDA]         rx-9070xt [ROCm/WSL2 or wgpu]
```

Why this shape:
- **No inbound ports at home.** Workers dial out, so home IP addresses and machines are never reachable by players.
- **Workers can go offline** (reboots, personal use of the 2080 box) without breaking anything. The coordinator marks them offline and the UI shows it.
- **One place for accounts, quotas and storage.** Workers stay stateless apart from a scratch directory and a checkpoint cache.

Hosting the coordinator, two options:
1. A small always-on VPS. Simplest and most reliable. Low CPU needs because it only relays.
2. On the DGX Spark at home, exposed through **Cloudflare Tunnel**. No VPS cost, no open ports, but the whole service is down whenever that machine or the home connection is.

Recommendation: start with option 2 for the invite-only phase, move to option 1 if availability matters.

Workers and the browser share the same `rl-envs`, `rl-agents` and `rl-training` crates. There is never a second implementation of an environment.

## 4. Accounts and authentication

- Players sign in with **OAuth** (GitHub and/or Google) or passkeys. The project never stores passwords.
- Sign-in succeeds only for accounts holding an invite code, or on an allow-list.
- Invite codes belong to a **group** (e.g. "friends", "RL class 2027"). A code can be single-use or multi-use with an expiry, so a whole class can join with one code.
- Quotas can be set per group, e.g. a class gets more GPU-minutes during an assignment week.
- The coordinator issues short-lived session tokens. The browser uses them for the API and WebSocket.
- Each worker has its own long-lived machine token. Tokens can be revoked individually from the coordinator.
- Roles: `player`, `admin`. Admins see all queues, can cancel any job, and can drain a worker.
- Privacy: the coordinator stores an email or OAuth ID, display name, experiments and policies. Publish a short privacy notice and support account deletion. GDPR applies if players are in the EU.

## 5. Choosing where to train

The Train view shows a **server directory** from the coordinator:

```text
SERVER        BACKEND        STATUS    QUEUE   EST. WAIT   YOUR QUOTA LEFT
dgx-spark     CUDA · GB10    online    2       ~4 min      38 GPU-min
rtx-2080      CUDA · 8 GB    online    0       now         38 GPU-min
rx-9070xt     ROCm · 16 GB   offline   —       —           —
local         WebGPU         ready     —       now         unlimited
```

- The player picks a server. "Auto" picks the shortest estimated wait among compatible servers.
- If the coordinator is unreachable, or every compatible server is offline, the client falls back to **local WebGPU**, then **local WASM CPU**. The player is told and can accept or wait.
- Tabular jobs (Levels 1–6) default to local, because they finish faster than a network round trip.
- The current target is always visible in the shell, e.g. `TRAINING ON: dgx-spark [CUDA · GB10]`.

A player can also run `calypso-worker` in **standalone mode** on their own machine. It then serves only that player on `localhost`, with no coordinator, accounts or quotas.

## 6. Scheduling, quotas and fairness

| Mechanism | Rule |
|---|---|
| Slots per device | Each worker advertises N concurrent job slots. Small RL networks under-use a GPU, so e.g. Spark ≈ 8 slots, 9070 XT ≈ 3, 2080 ≈ 2. Tuned by measurement |
| Queue order | Round-robin across players, FIFO within a player. One heavy user cannot starve others |
| Per-player limits | Max 1–2 running jobs and max K queued jobs per player |
| Quota | Daily GPU-minutes per player. Admins can grant more |
| Hard job limits | Max wall-clock time and max environment steps per job, enforced by the worker |
| Level budgets | A job for a campaign level must fit that level's training budget and model-size limit. The coordinator rejects it otherwise |

The in-game **compute budget mechanic** and the **server quota** are deliberately the same idea. The quota then feels like part of the game rather than a restriction.

## 7. Security rules

1. **Job configs are pure data, never code.** Rewards, observations and opponents are chosen from parameterised components. No player-supplied scripts ever run on a worker. If custom reward code is ever wanted, it runs only locally in the browser.
2. The coordinator validates every config against a versioned schema, plus the bounds in section 6.
3. Uploaded policies are weight files in one fixed format with size limits. They are deserialised into known architectures only.
4. Workers run as an unprivileged user, ideally in a container with GPU access, and only reach the coordinator.
5. Rate limits apply on sign-in, job submission and WebSocket connections.

## 8. Job protocol

The Milestone 2 **experiment configuration is the job specification**: `ExperimentConfig`, schema 1, in
`crates/rl-training/src/experiment.rs`.

```text
schema       1
level        { id, hash }              hash = FNV-1a of the level file; a changed level refuses the job
edits        { rewards, observation }  checked against the level's player_editable rules (EditMode::Strict)
seed         training episode k sails on seed + k
algorithm    QLearning { alpha, gamma, epsilon }
episodes     at most the level's training budget
evaluation   { first_seed, episodes }  greedy test voyages, recorded as seed + action log
```

`ExperimentConfig::prepare` reports every problem with its field, and is the check a worker runs before accepting a
job. A run (`ExperimentRun`) records the curve (return, length, arrival, exploratory moves per episode), the test
voyages, the spec hash, the engine version and the trained policy; `fingerprint()` hashes all of it, so a rerun can be
compared bit for bit.

Browser ↔ coordinator:

```text
GET  /v1/servers                  directory with status, queue, estimated wait
POST /v1/jobs                     { server | "auto", ExperimentConfig } → job_id
GET  /v1/jobs                     the player's jobs
GET  /v1/jobs/{id}                status, position in queue
WS   /v1/jobs/{id}/stream         metrics, evaluation results, logs, checkpoint notices
POST /v1/jobs/{id}/cancel
GET  /v1/jobs/{id}/policy         final/best policy weights
GET  /v1/jobs/{id}/replays/{n}    evaluation replay (seed + action log)
GET  /v1/policies                 the player's policy library
```

Worker ↔ coordinator, over one outbound WebSocket per worker:

```text
worker → hello { name, backends, device, memory, slots, version }
coord  → assign { job_id, ExperimentConfig, opponent policies }
worker → progress / metrics / checkpoint / done / failed
coord  → cancel { job_id }
```

Job states: `queued → assigned → running → completed | cancelled | failed | lost`. `lost` means the worker disappeared. The job is re-queued once from its last checkpoint, then failed.

Storage:
- The **policy library lives on the coordinator**, per account, so it follows the player across devices. The browser caches it in IndexedDB.
- The Ithaca finale needs each player's historical policies. The server-side library provides them.
- Metrics are stored per job, so a player who closes the tab sees the full curve when returning.
- Retention limits apply per account, e.g. the last N experiments plus every policy the player pins.
- **Replays are never deleted by retention while they are part of a player's offline dataset.** Levels 13 and 16 depend on the player's whole campaign history. Seed + action-log replays are tiny, so keeping all of them is cheap.

## 9. Reproducibility on GPUs

- Environment simulation runs on CPU with integer or fixed-point state, so it is bit-exact everywhere.
- GPU training is not bit-reproducible across devices, and sometimes not across runs on one device.
- Every experiment records `backend`, `device`, `framework version` and `worker name`.
- Guarantee: same seed + config + device + version ⇒ same result where the backend allows it. Evaluating a fixed policy on fixed seeds is always deterministic, which is what campaign scoring uses.

## 10. Player experience while queued

- Jobs run without the browser open. The player can keep playing, for example manually or with an earlier policy, and is notified when training finishes.
- Estimated wait is shown before submitting, so choosing a server is a meaningful decision.
- Live curves stream while running. If the queue is long, the UI suggests local training for small jobs.

## 11. When to build it

| Milestone | Work |
|---|---|
| 0–1 | Training runs in a Web Worker from day one. Introduce a `TrainingTarget` abstraction with only the local implementation. |
| 2 | Freeze `ExperimentConfig` as a versioned, serialisable job spec with a validation schema. *Done: schema 1 in `crates/rl-training/src/experiment.rs`.* |
| After 1 | Burn hardware spike on all three servers + browser WebGPU. On the 9070 XT, test both WSL2 + ROCm and native Windows + wgpu. Browser and developer Mac done 2026-11-04 (§2); servers pending. |
| **7.5a** | `calypso-worker` in standalone mode with **tabular Q-learning**. Proves streaming, cancellation and policy download. |
| **7.5b** | `calypso-coordinator`: OAuth sign-in, invite list, server directory, per-device queues, quotas, policy library. Connect all three workers. |
| 8 | DQN on workers. Regression test: DQN matches tabular on an already-solved level, on every backend. |
| Before Level 13 | **Dataset jobs.** The coordinator builds offline datasets by re-simulating stored replays (seed + action log) and ships them to workers. Small decision transformers, a few layers with a context of 20–50 steps, fit on every server including the 8 GB RTX 2080. Inference runs in the browser via wgpu. |
| Before the first class | Load test with a class-sized burst of jobs, privacy notice, backups of the coordinator database, basic monitoring. |
