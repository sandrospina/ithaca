# Reinforcement Odyssey: The Road to Ithaca — Development Plan

Revision 2, 2026-11-02. Revision 1 is archived in `docs/archive/development-plan-v1-original.md`.

This plan is the single entry point. Detailed designs live in:

| Document | Covers |
|---|---|
| `docs/campaign.md` | Odyssey frame story, 17-level map, algorithms by level, challenge format |
| `docs/tutorials.md` | Per-level tutorial system, explanation depths, classroom support |
| `docs/environment-model.md` | Data-driven levels, world engines, components, rewards, observations |
| `docs/training-architecture.md` | Shared GPU servers, coordinator and workers, local fallback, accounts |
| `docs/visual-style.md` | Hybrid phosphor-terminal aesthetic and visual encoding rules |

------------------------------------------------------------------------

## 0. Decision Log

| Date | Decision | Detail in |
|---|---|---|
| 2026-11-02 | Title: **Reinforcement Odyssey: The Road to Ithaca**. Web search found no conflicting game or trademark; no formal register search yet | `campaign.md` |
| 2026-11-02 | Odyssey framing: a journey from Ogygia to Ithaca. Calypso is mentor and narrator | `campaign.md` |
| 2026-11-02 | Aesthetic: vector-phosphor arena inside an amber mainframe-terminal UI shell | `visual-style.md` |
| 2026-11-02 | Every RL concept has one fixed shape and colour across all levels | `visual-style.md` |
| 2026-11-02 | Training on the developer's GPU servers through a central coordinator; players pick a server; fallback to local WebGPU, then local CPU | `training-architecture.md` |
| 2026-11-02 | Multiple users, invite-only: friends, students, classes. Public access out of scope | `training-architecture.md` |
| 2026-11-02 | Deep-learning framework: **Burn** (CUDA, ROCm, Vulkan and browser WebGPU from one model definition). Candle rejected: no ROCm, no WebGPU | `training-architecture.md` |
| 2026-11-02 | RX 9070 XT machine stays on Windows 11: test ROCm under WSL2 and native wgpu, keep the better one | `training-architecture.md` |
| 2026-11-02 | New Level 13: offline RL with behaviour cloning and decision transformers. Campaign is now Levels 0–16 | `campaign.md` |
| 2026-11-02 | Every level ships with a tutorial; tutorials are data files tested in CI | `tutorials.md` |
| 2026-11-02 | Levels are data files composing coded mechanics. No scripting language in levels, jobs or tutorials | `environment-model.md` |

------------------------------------------------------------------------

## 1. Project Goal

Build a web-based game that teaches reinforcement learning (RL) through progressively harder adversarial environments, framed as Odysseus' voyage home.

The central design principle is:

> Each level introduces one or two new things that the agent must observe, model, predict, or control.

Difficulty arises from increasingly demanding RL problems: larger state spaces, stochasticity, partial observability, memory, continuous control, adaptive opponents, offline data, multi-agent interaction and communication. It never comes merely from enemies with more health or speed.

The player repeatedly follows this loop:

**Understand → Configure → Train → Analyse → Deploy → Fight → Improve**

"Understand" is delivered by a per-level tutorial (`docs/tutorials.md`).

Audience: invited friends, students and classes. Teachers are a first-class user role.

The project is built by a **single developer**, so development proceeds vertically: first a very small end-to-end system, then incremental expansion.

------------------------------------------------------------------------

## 2. Technology Stack

### Core simulation and game logic — Rust

Rust is used for environment simulation, level loading and compilation, rewards, observations, deterministic seeded simulation, scripted opponents, training, replay, evaluation and the server components.

Crates:

-   `serde`, `serde_json`, `ron` — level files, configs, wire formats;
-   deterministic RNG: xoshiro256** implemented in `rl-core`, so no dependency upgrade can change replay streams;
-   `fixed` — fixed-point arithmetic for the continuous Sea engine;
-   `rayon` — parallel CPU simulation on workers;
-   `tokio`, `axum` — coordinator and worker services;
-   `tracing` — diagnostics;
-   `wasm-bindgen`, `web-sys` — browser interface, Web Workers;
-   **`burn`** — neural networks on every backend (decided);
-   `proptest` — property tests.

Tabular RL is implemented directly in Rust without a framework.

### Web frontend

-   **Rust + Leptos** for the application (kept; charts and KaTeX via small JavaScript interop);
-   WebAssembly for browser-side simulation and training, always in a **Web Worker**;
-   Canvas2D first for the arena, WebGL2 later for bloom;
-   Style: see `docs/visual-style.md`.

### Training architecture

Three execution modes, one shared codebase:

1.  **Browser, local** — tabular levels by default; small networks via WebGPU, or WASM CPU as last resort.
2.  **Shared GPU servers** — `calypso-worker` on each GPU machine, connected outbound to a `calypso-coordinator` that players use.
3.  **Standalone worker** — a player runs `calypso-worker` on their own machine for themselves.

Hardware:

| Machine | GPU | Backend |
|---|---|---|
| DGX Spark (ARM64, assumed Linux) | Blackwell GB10, 128 GB unified | CUDA |
| RTX 2080 PC (assumed Linux) | 8 GB | CUDA |
| RX 9070 XT PC (Windows 11) | 16 GB | ROCm in WSL2, or native wgpu |

Rendering is never required for training. Full design: `docs/training-architecture.md`.

------------------------------------------------------------------------

## 3. Repository Structure

``` text
RL-Game/
├── Cargo.toml                      workspace
├── adversarial_rl_game_development_plan.md
├── docs/
│   ├── campaign.md
│   ├── tutorials.md
│   ├── environment-model.md
│   ├── training-architecture.md
│   ├── visual-style.md
│   └── archive/
├── content/
│   ├── levels/                     RON level files, one per level
│   └── tutorials/                  tutorial text + step scripts per level
├── crates/
│   ├── rl-core/                    World trait, specs, observation features, reward terms, metrics
│   ├── rl-envs/                    Grid engine, Sea engine, components, behaviours, generators, level compiler
│   ├── rl-agents/                  random, Q-learning, DQN, PPO, BC, decision transformer
│   ├── rl-training/                trainer, evaluator, replay, experiment config, datasets
│   ├── rl-tutorial/                tutorial step engine, lesson-mode tracing
│   ├── rl-web/                     Leptos UI, renderer, theme tokens, workers
│   ├── calypso-worker/             native training worker
│   └── calypso-coordinator/        accounts, queues, quotas, storage, server directory
└── tests/
```

`rl-core` stays independent of graphics, UI, networking and ML frameworks.

------------------------------------------------------------------------

## 4. Core Environment API

The environment is multi-agent from day one. A single-agent level is a world with one learning agent. Full design: `docs/environment-model.md`.

``` rust
pub trait World {
    fn reset(&mut self, seed: u64) -> Observations;          // one per agent
    fn step(&mut self, actions: &JointAction) -> StepResults;
    fn action_mask(&self, agent: AgentId) -> ActionMask;      // fixed size, no allocation
    fn spec(&self) -> &WorldSpec;                             // observation + action shapes
}
```

Each agent's step result follows Gymnasium's vocabulary:

``` rust
pub struct StepResult<O> {
    pub observation: O,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub info: StepInfo,
}
```

A Gymnasium-style single-agent wrapper sits on top for simple levels and interoperability.

Key design points:

-   **Levels are data.** A RON file composes entities, components, behaviours, reward terms and observation features from a fixed library, then is validated and compiled into flat arrays before running.
-   **Rewards** are weighted sums of named terms. **Observations** are ordered lists of named features with known sizes.
-   **Opponents** are `PolicyProvider`s: scripted behaviours and trained policies plug in the same way.
-   **Policies store the hash of their observation and action specs** and refuse to load into incompatible worlds.
-   **Two engines:** Grid (integer state, Levels 0–6) and Sea (fixed-point continuous 2D, Levels 7–16).

### Non-negotiable requirements

-   deterministic seeds, bit-exact across native and WASM;
-   headless execution;
-   fast reset and step;
-   serialisable configuration;
-   episode recording as seed + action log;
-   reproducible evaluation;
-   interchangeable agents and opponents.

------------------------------------------------------------------------

# 5. Development Milestones

Each level milestone includes its **level file**, its **tutorial** (`docs/tutorials.md`), and passing CI checks for both. A level is not done until its tutorial's guided experiments pass.

## Milestone 0 — Project skeleton and technical spike

**Goal:** prove that data-driven Rust simulation can drive a browser visualisation in the chosen style.

Build:

-   Cargo workspace;
-   `rl-core` with the multi-agent `World` trait;
-   Grid engine, `LevelDef`, RON loading, validation and compilation;
-   Level 0, **Calypso's Loom**, as a data file;
-   minimal Leptos application with the amber terminal shell;
-   Rust/WASM bridge and a Web Worker running the simulation;
-   phosphor Canvas2D renderer using theme tokens;
-   **style-sheet scene** showing every glyph and colour token;
-   keyboard-controlled agent;
-   Level 0 tutorial (text beats, HUD labelling the RL loop).

No RL yet.

### Exit criteria

You can open the application, play Level 0 by hand, reset with a seed, and run the same level file headlessly from a Rust test.

------------------------------------------------------------------------

## Milestone 1 — Level 1: The Raft

**RL topic:** tabular Q-learning.

Environment: 5×5 grid, raft, shore (goal), four discrete actions, deterministic transitions. State: `(x, y)`. Rewards: goal +10, step −1.

Implement:

-   random agent;
-   Q-learning agent with ε-greedy exploration, α and γ;
-   training inside a Web Worker behind a `TrainingTarget` abstraction (local only for now);
-   episode runner;
-   Q-table persistence (browser IndexedDB);
-   reward and observation **catalogues**, even with only two entries each;
-   **tutorial engine**: step script format, the seven lesson beats, depth toggle;
-   **live Q-learning walkthrough**: executing pseudocode with live values;
-   episode training budget and star rating.

### Web UI

``` text
CONFIGURE → TRAIN → ANALYSE → DEPLOY
```

Display the arena, current episode, cumulative reward, training progress, Q-values for a selected cell and policy arrows.

### Exit criteria

A player completes the Level 1 tutorial, trains an agent in the browser, and watches the learned policy reach the shore.

------------------------------------------------------------------------

## Spike — Burn on all hardware (after Milestone 1)

A one-to-two-day hardware test, not deep-RL work:

-   a small Burn training example on the DGX Spark (CUDA, ARM64, Blackwell), the RTX 2080 (CUDA) and the RX 9070 XT (ROCm in WSL2 and native wgpu);
-   the same model running in the browser via WebGPU.

### Exit criteria

Each machine has a confirmed working backend, recorded in `docs/training-architecture.md`.

Status (2026-11-04): the spike program is in `spikes/burn/`. Browser WebGPU and the developer's Mac pass; browser
training needs kernel autotuning off (details in `training-architecture.md` §2). The three servers are still to run.

------------------------------------------------------------------------

## Milestone 2 — RL experiment infrastructure

**Goal:** build the infrastructure every later level needs.

Experiment configuration, frozen as a **versioned, validated job specification** reused by the servers later:

``` text
level file reference + player edits
seed
algorithm
hyperparameters
training budget
evaluation episodes
```

Collect episode return, episode length, success rate, exploration rate, training steps and wall-clock time.

Implement experiment save/load, policy save/load with spec hashes, replay recording as seed + action log, replay playback, deterministic evaluation, run comparison, headless tutorial tests in CI, and the concept library.

### Exit criteria

Two experiments with different hyperparameters can be run and compared reproducibly, and every tutorial runs headlessly in CI.

Status (2026-11-05): done.
- `ExperimentConfig` (schema 1, `crates/rl-training/src/experiment.rs`) is the job specification: level id and file
  hash, player edits, seed, algorithm with hyperparameters, training budget, evaluation seeds. It is validated
  strictly against the level's editing rules; the same check will run on the servers.
- A run records per-episode return, length, arrival and exploratory moves, the training steps, the test voyages as
  seed + action log, the engine version and a fingerprint. The browser measures wall-clock time.
- The COMPARE view runs, reruns (checking the fingerprint), adopts, compares up to four runs and replays test
  voyages. The log is kept per level in IndexedDB. Tutorial guided experiments go through the same path.
- Saved rafts carry the observation and the spec hash. CI runs every tutorial step headlessly: highlights must
  exist in their level and every guided experiment must show its lesson.
- The concept library (`content/concepts.ron`) opens from LIBRARY and generates `docs/rl-concepts.md`.

------------------------------------------------------------------------

## Milestone 3 — Level 2: The Lotus-Eaters

**RL topic:** reward engineering.

Add walls, hazards, lethal cells, alternative paths and the **lotus lure**: a small, repeated reward that traps badly rewarded agents. Expose the reward weights to the player through the reward editor.

``` text
goal       +100
death      -100
step         -1
hazard       -5
lotus        +1 per step   (the trap)
```

### Exit criteria

At least two deliberately bad reward configurations produce recognisably pathological policies, one of them the agent that never leaves the lotus field.

------------------------------------------------------------------------

## Milestone 4 — Level 3: The Cyclops' Cave

**RL topic:** state representation and adversarial behaviour.

Add Polyphemus, a scripted `chase_shortest_path` opponent, implemented as a `PolicyProvider`.

The inadequate default observation is the agent's position only. The **Observation Builder** lets the player add:

``` text
[x] agent position
[ ] Cyclops position
[ ] Cyclops direction
[ ] previous action
```

The UI shows the resulting Q-table size for each choice.

### Teaching objective

> A policy cannot learn to react to information that is absent from its observation.

### Exit criteria

The player demonstrates experimentally that adding the Cyclops to the observation substantially improves performance. The Level 1 agent visibly fails at the start of the level.

**This milestone completes the first playable prototype. Playtest it with a few friends or students before continuing.**

Status (2026-11-05): built. Polyphemus starts in one of two tunnels, picked by the seed, and chases by shortest path
(`chase_shortest_path`, a `PolicyProvider` over a precomputed move table). Pinned by
`crates/rl-training/tests/cyclops_cave.rs`: a raft that observes only its position escapes at most 7 of the 10 test
seeds (the objective needs 8); with Polyphemus' direction (245 rows) it escapes all 10 within 500 episodes; with his
position (2,401 rows) at least 8 within 2,000. The level's comparison task completes from the experiment log.

------------------------------------------------------------------------

## Milestone 5 — Level 4: The Bag of Winds

**RL topic:** stochastic transitions and MDPs.

Add wind regions with configurable transition noise and a stochastic Cyclops.

``` text
requested NORTH

80% NORTH
10% WEST
10% EAST
```

Visualise repeated outcomes from the same state and action.

### Exit criteria

The player compares deterministic and stochastic training and sees why a policy must account for expected outcomes, for example the safe path beating the short one.

------------------------------------------------------------------------

## Milestone 6 — Level 5: The House of Hades

**RL topic:** partial observability.

Restrict observations to local view, coarse shade direction, visible hazards and selected sensors. The debug view shows true state and agent observation side by side ("lifting Calypso's veil").

### Exit criteria

The same world produces policies of different performance depending on the observation configuration, and the debug view shows aliased states.

------------------------------------------------------------------------

## Milestone 7 — Level 6: Circe's Instructions

**RL topic:** memory.

Circe's beacons are visible only at the start. Add switches, doors and delayed objectives. Support frame stacking, previous action and previous reward as observation features. Recurrent policies come later, in Level 9.

### Exit criteria

A memoryless policy demonstrably underperforms a frame-stacked one.

------------------------------------------------------------------------

## Milestone 7.5a — Standalone training worker

`calypso-worker` in standalone mode running **tabular Q-learning** jobs from the Milestone 2 job specification. Proves streaming metrics, cancellation, policy download and the `TrainingTarget` remote path.

### Exit criteria

A browser client trains Level 3 on a worker and on local hardware with identical results.

## Milestone 7.5b — Coordinator

`calypso-coordinator`: OAuth sign-in, invite codes and groups, server directory, per-device queues, round-robin fairness, quotas, policy library, teacher role, quizzes and progress view. Connect all three workers. Hosted at first on the DGX Spark behind Cloudflare Tunnel.

### Exit criteria

Two invited players train concurrently on different servers, and a teacher sees both players' progress.

------------------------------------------------------------------------

# 6. Deep RL Milestones

## Milestone 8 — Neural network training infrastructure

Pipeline:

``` text
level → parallel simulation workers → experience collection
      → replay/rollout storage → Burn network → optimiser → checkpoint
```

Implement tensor conversion from observation specs, checkpoints, training on every backend, cancellation, periodic evaluation, policy export and in-browser inference. Add the deep-RL walkthrough grain: transition into buffer, minibatch, gradient step, target sync.

First implement **DQN** only.

### Exit criteria

DQN solves a level already solved by tabular Q-learning, on every backend, as a regression test.

------------------------------------------------------------------------

## Milestone 9 — Level 7: The Wine-Dark Sea

**RL topic:** function approximation and DQN.

First level on the **Sea engine**: fixed-point continuous 2D, discrete steering actions, currents, islands, reefs and a harbour. Before this milestone, re-express one grid level in the Sea engine as a regression test.

Observation: heading, velocity, ray sensors, harbour bearing.

Teach the move from `Q[state][action]` to `Qθ(state, action)`.

### Exit criteria

DQN learns a useful policy. The UI shows loss, return and evaluation success, and the tabular memory counter shows why the table fails.

------------------------------------------------------------------------

## Milestone 10 — Level 8: Scylla and Charybdis

**RL topic:** policy gradients and actor-critic.

Continuous throttle and rudder in `[-1, 1]` through the strait, with Scylla's strike zone and Charybdis' vortex. Implement **PPO**. SAC only if a level later shows a clear need.

### Exit criteria

The player trains and deploys a continuous-control policy.

------------------------------------------------------------------------

## Milestone 11 — Level 9: The Cattle of Helios

**RL topic:** delayed consequences, resources and recurrent policies.

Add crew resources (food, health), pickups, and the sacred cattle: a locally attractive lure with a delayed catastrophic penalty. Add **recurrent PPO** (LSTM or GRU). This also revisits Level 6 with learned memory.

### Exit criteria

Successful policies manage at least two limited resources while pursuing the objective, and short-horizon γ visibly eats the cattle.

------------------------------------------------------------------------

# 7. Adversarial Campaign

## Milestone 12 — Level 10: The Phaeacian Games

**RL topic:** robustness and training distributions.

Opponent archetypes from the scripted behaviour library: random, aggressive, defensive, patrol, ambush, evasive. Train against one, a random mix, or a weighted mix.

``` text
training distribution ≠ deployment distribution
```

### Exit criteria

A policy overfitted to one opponent loses to another, while a policy trained on the mixture performs better across the set.

------------------------------------------------------------------------

## Milestone 13 — Level 11: Wrestling Proteus

**RL topic:** self-play.

Both sides are trainable. Keep policy snapshots and train against a league of historical checkpoints. Track win rate, draw rate, Elo-style rating and performance against history.

### Exit criteria

An agent improves through self-play without forgetting how to beat older strategies.

------------------------------------------------------------------------

## Milestone 14 — Level 12: Poseidon's Wrath

**RL topic:** domain randomisation and adversarial environments.

Level generators with parameter distributions: obstacles, spawns, resources, sensor noise, currents. A scripted Poseidon chooses parameters to minimise the agent's return, within the same bounds.

``` text
π_agent vs π_environment
```

### Exit criteria

Policies trained under randomisation generalise better to unseen seas than policies trained on one fixed sea.

------------------------------------------------------------------------

## Milestone 15 — Level 13: The Tale at Alcinous' Hall (new)

**RL topic:** offline RL, behaviour cloning, decision transformers.

Revisits the Level 7 world. The dataset is the player's own recorded voyages there, rebuilt from seed + action-log replays: good and bad checkpoints, plus manual voyages.

Implement:

-   dataset jobs on the coordinator and workers;
-   behaviour cloning;
-   a small **decision transformer** (few layers, context of 20–50 steps), trainable on every server and runnable in the browser;
-   the **return-to-go dial** in the UI;
-   comparison with PPO under the same budget;
-   opt-in shared datasets for classes.

### Exit criteria

Behaviour cloning visibly copies mistakes. The decision transformer's behaviour quality follows the return-to-go dial, and fails above the best return in the data.

------------------------------------------------------------------------

# 8. Multi-Agent Campaign

Levels 14–16 share one **Ithaca arena spec**, so their matches can feed the Level 16 dataset.

## Milestone 16 — Level 14: Father and Son

**RL topic:** multi-agent RL.

Odysseus and Telemachus against the suitors' scouts, 2v2. Start with shared-parameter PPO, then centralised-critic PPO (MAPPO). Compare with independent learners.

### Exit criteria

Cooperative behaviour emerges that a single agent cannot achieve.

------------------------------------------------------------------------

## Milestone 17 — Level 15: The Silent Signal

**RL topic:** learned communication.

Physical action plus a message `∈ {0..15}`, shown as Greek glyphs during replay. Limited sensor range makes communication useful.

### Exit criteria

Communication measurably improves team success, and the replay shows what glyphs come to mean.

------------------------------------------------------------------------

## Milestone 18 — Level 16: Ithaca

The final arena combines partial observability, memory, rich control, resources, stochasticity, multiple agents, adaptive opponents and communication.

The **suitors** are decision transformers trained on the player's own recorded matches from Levels 14–15, then fine-tuned by self-play against the player's policy library. They play like the player and exploit the player's habits.

### Exit criteria

Policies from multiple training approaches compete under a standard evaluation protocol, and the suitors measurably exploit weaknesses in the player's historical policies.

------------------------------------------------------------------------

# 9. RL Debugger

Built incrementally from Milestone 1. It is core gameplay, not developer tooling.

Expose: episode return, success or win rate, episode length, Q-values, TD error, network loss, policy entropy, exploration rate, action distribution, training throughput, inference time, observation vs true state, replay timeline, and the **live pseudocode walkthrough** from the tutorials.

Replay supports pause, step forward and back, and inspecting observation, action, reward and policy output.

------------------------------------------------------------------------

# 10. Compute Budget Mechanic

-   **From Level 1:** every level has an episode budget and star ratings (`docs/campaign.md`). Brute force is never the universal answer.
-   **From Level 7:** the full budget adds environment steps, training time and optional model-size limits.
-   **Server quotas use the same idea.** Daily GPU-minutes per player, set per group, feel like part of the game.

``` text
score = performance - compute_penalty - model_complexity_penalty
```

The formula stays configurable per level file.

------------------------------------------------------------------------

# 11. Opponent Escalation

``` text
Random opponent
      ↓
Deterministic scripted opponent          (Cyclops, Level 3)
      ↓
Stochastic opponent                      (Level 4)
      ↓
Strategy-switching opponents             (Phaeacian Games, Level 10)
      ↓
Learning opponent / self-play            (Proteus, Level 11)
      ↓
Adversarial environment                  (Poseidon, Level 12)
      ↓
Multi-agent opponents                    (Levels 14–15)
      ↓
Opponents that imitate and exploit you   (Suitors, Level 16)
```

An opponent is always a `PolicyProvider`, never special-case game logic.

------------------------------------------------------------------------

# 12. Historical Policy and Final Adversary

Selected player policies are saved throughout the campaign as named crew members in the server-side policy library. Replays are kept as seed + action logs and never deleted while part of a dataset.

At Ithaca the suitors are built from this history:

1.  a decision transformer trained on the player's recorded matches in the Ithaca arena;
2.  fine-tuned by self-play against the player's saved policies.

This demonstrates adversarial training and policy exploitation, and Level 13 has already taught the technique.

------------------------------------------------------------------------

# 13. Web Application Layout

Amber terminal shell around a phosphor arena (`docs/visual-style.md`).

``` text
+-----------------------------------------------------------+
| CAMPAIGN | LEVEL | TRAINING ON: dgx-spark [CUDA] | LIBRARY |
+----------------------------+------------------------------+
|                            | Calypso / tutorial panel     |
|          ARENA             |------------------------------|
|      (phosphor vector)     | Observation Builder          |
|                            | [x] position                 |
|                            | [ ] Cyclops position         |
+----------------------------+------------------------------+
| Oscilloscope graphs: return, success, loss, entropy        |
+-----------------------------------------------------------+
| CONFIGURE       TRAIN       ANALYSE       DEPLOY          |
+-----------------------------------------------------------+
```

Dedicated views, added over time: campaign map (vector Aegean chart), server directory, experiment comparison, replay debugger, concept library, policy library, self-play league, teacher dashboard, tournament arena.

------------------------------------------------------------------------

# 14. Testing Strategy

### Unit tests

Transitions, rewards, terminal conditions, observation features, deterministic seeds, action masks.

### Property tests

-   agents never leave valid bounds;
-   resources never exceed maximum;
-   identical seed + actions produce identical trajectories, natively and in WASM.

### Content tests

-   every level file loads, validates and compiles in CI;
-   random-action property tests run on every level;
-   every tutorial script runs headlessly and its guided experiments show the intended outcome.

### RL regression tests

Tiny environments with known optimal policies, e.g. a 2×2 grid. Q-learning, DQN and PPO are tested against them; DQN on every backend.

### Visual regression

The style-sheet scene is screenshotted in CI.

### Performance benchmarks

Environment steps per second, reset time, training throughput, WASM vs native, worker throughput per device. Measure simulation before optimising rendering.

------------------------------------------------------------------------

# 15. Development Order

``` text
Grid engine + level files
      ↓
browser renderer (phosphor style) + manual control
      ↓
Q-learning in a Web Worker
      ↓
tutorial engine + training UI
      ↓
metrics, replay, experiment system
      ↓
Level 2 → Level 3            → playtest first release
      ↓
stochasticity → partial observability → memory
      ↓
standalone worker → coordinator
      ↓
DQN → Sea engine → PPO → recurrent PPO
      ↓
robustness → self-play → adversarial environment
      ↓
offline RL + decision transformer
      ↓
multi-agent → communication → Ithaca
```

Every milestone leaves the project runnable.

------------------------------------------------------------------------

# 16. First Development Sprint

The immediate objective is the smallest vertical slice proving the architecture, not the game.

1.  **Create workspace:** `rl-core`, `rl-envs`, `rl-agents`, `rl-training`, `rl-tutorial`, `rl-web`.
2.  **Implement the Grid engine and level files:** width, height, map, spawn, goal, walls, seed; `LevelDef` → validate → compile; `reset()` and `step()`.
3.  **Test headless simulation:** thousands of steps without a renderer; seed determinism tests.
4.  **Compile to WASM in a Web Worker:** expose reset, step and state to the UI thread.
5.  **Render in the phosphor style:** grid, raft, shore, walls; style-sheet scene; amber shell. Keyboard movement. This is Level 0.
6.  **Implement Q-learning:** simple and readable, with lesson-mode trace events.
7.  **Add training controls:** episodes, α, γ, ε, seed, and the TRAIN button, all running in the worker.
8.  **Visualise the learned policy:** arrows, Q-values, live pseudocode walkthrough.
9.  **Add deployment:** RUN LEARNED POLICY animates an evaluation episode.
10. **Save the experiment:** level reference, hyperparameters, Q-table, metrics, seed.
11. **Write the Level 0 and Level 1 tutorials** as data files.

------------------------------------------------------------------------

# 17. First Major Target

Contains only:

-   **Level 0** — Calypso's Loom, manual play;
-   **Level 1** — The Raft, Q-learning;
-   **Level 2** — The Lotus-Eaters, reward engineering;
-   **Level 3** — The Cyclops' Cave, observation expansion;
-   their tutorials and the concept library;
-   training, metrics, replay and comparison;
-   Configure, Train, Analyse and Deploy;
-   browser-only training. No servers.

Do **not** start servers or DQN before this release has been playtested.

Level 3 is the critical validation point:

> The old agent fails because the world now contains information that its existing representation does not model.

If this transition is understandable and enjoyable for real players, continue with the roadmap.

------------------------------------------------------------------------

# 18. Second Major Target

Levels 4–7: stochastic transitions, partial observability, memory, standalone worker, coordinator with accounts and quotas, Burn, DQN, the Sea engine, GPU training on all three machines. First class or group of invited players.

# 19. Third Major Target

Levels 8–10: PPO, recurrent PPO, resource management, scripted opponent families, robustness evaluation.

# 20. Fourth Major Target

Levels 11–13: trainable opponents, self-play leagues, adversarial environment generation, offline RL and decision transformers, historical policy evaluation.

# 21. Final Major Target

Levels 14–16: 2v2, shared and centralised-critic team policies, communication, the decision-transformer suitors, league and tournament evaluation.

------------------------------------------------------------------------

# 22. Scope Rules

1.  **Simulation before graphics.**
2.  **One new RL concept per level where possible.**
3.  **No deep RL until tabular gameplay is fun and understandable.**
4.  **No self-play until single-agent training is reliable.**
5.  **No MARL until self-play is reliable.**
6.  **Every environment must run headlessly.**
7.  **Every experiment must be reproducible from a seed and configuration.**
8.  **Rendering must never be required for training.**
9.  **Educational visualisation is more important than graphical fidelity.**
10. **Do not implement an algorithm unless a gameplay problem motivates it.**
11. **Levels, jobs and tutorials are data, never code.** No scripting language anywhere a player or teacher can write.
12. **Every level ships with a tutorial whose guided experiments pass CI.**
13. **No server work before the first major target has been playtested.**
14. **Every visual element uses a theme token and its fixed visual encoding.**

------------------------------------------------------------------------

# 23. Immediate Backlog

-   [x] Initialise git repository.
-   [x] Create Cargo workspace.
-   [x] Create `rl-core`: `World`, `WorldSpec`, `JointAction`, `StepResult`, `ActionMask`, `PolicyProvider`.
-   [x] Implement seeded RNG.
-   [x] Implement Grid engine and `LevelDef` (RON load, validate, compile).
-   [x] Write Level 0 and Level 1 level files.
-   [x] Add environment unit and property tests.
-   [x] Add headless episode runner.
-   [x] Implement random agent.
-   [x] Implement Q-learning agent.
-   [x] Add Q-learning regression test.
-   [x] Create Leptos app with amber terminal shell and theme tokens.
-   [x] Run simulation and training in a Web Worker.
-   [x] Phosphor Canvas2D renderer and style-sheet scene.
-   [x] Keyboard/manual control (Level 0).
-   [x] Tutorial engine and Level 0–1 tutorials: text at three depths, highlights, predictions and guided experiments checked in CI.
-   [x] Train button, reward/episode graph, Q-value and policy views.
-   [x] Live pseudocode walkthrough.
-   [x] Q-table persistence in IndexedDB, and the `TrainingTarget` abstraction (local worker only).
-   [x] Learned-policy deployment.
-   [x] Experiment serialisation and replay recording (Milestone 2: `ExperimentConfig`, COMPARE, replays from action logs).
-   [ ] Burn hardware spike on all three servers and the browser. Browser and Mac done; servers pending (`spikes/burn/README.md`).
-   [x] Level 2: hazards, lotus lure, reward editor, tutorial.
-   [x] Level 3: scripted Cyclops, Observation Builder, spec hashes, tutorial.
-   [x] Compare agent-only observation with agent + Cyclops (a Level 3 task, settled in COMPARE).
-   [ ] Playtest with a few friends or students and evaluate whether the loop is educational and enjoyable.

------------------------------------------------------------------------

# 24. Definition of Success for the First Release

A new player can:

1.  open the game in a browser;
2.  understand the objective without reading RL code;
3.  sail Level 0 by hand and name the parts of the RL loop;
4.  follow the Level 1 tutorial and its live Q-learning walkthrough;
5.  configure, train and watch a Q-learning agent learn;
6.  inspect its Q-values and policy;
7.  deploy the trained policy;
8.  break the agent with bad rewards in the Lotus-Eaters, and fix it;
9.  meet the Cyclops and watch the previous agent fail;
10. add the Cyclops to the observation, retrain, and see the improvement;
11. make a prediction before each experiment and see whether it was right.

At that point the project has demonstrated its central idea, and it is worth investing in servers, DQN, PPO, self-play, offline RL and multi-agent RL.

------------------------------------------------------------------------

# 25. Open Questions

-   Operating system of the DGX Spark and RTX 2080 machines (assumed Linux).
-   Open-source licence, relevant for classroom use.
-   Formal trademark search before any public use of the title.
-   Sound design: simple vintage beeps fit the style.
