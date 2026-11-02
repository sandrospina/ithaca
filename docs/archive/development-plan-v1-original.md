# Adversarial Reinforcement Learning Game --- Development Plan

## 1. Project Goal

Build a web-based game that teaches reinforcement learning (RL) through
progressively harder adversarial environments.

The central design principle is:

> Each level introduces one or two new things that the agent must
> observe, model, predict, or control.

The player should not progress simply by encountering enemies with more
health or speed. Difficulty should arise from increasingly demanding RL
problems: larger state spaces, stochasticity, partial observability,
memory, continuous control, adaptive opponents, multi-agent interaction,
and communication.

The player repeatedly follows this loop:

**Understand → Configure → Train → Analyse → Deploy → Fight → Improve**

The project is intended for a **single developer**, so development
should proceed vertically: first build a very small end-to-end system,
then expand the RL and game capabilities incrementally.

------------------------------------------------------------------------

## 2. Recommended Technology Stack

### Core simulation and game logic --- Rust

Use Rust for:

-   environment simulation;
-   state and action definitions;
-   reward calculation;
-   deterministic seeded simulations;
-   opponent logic;
-   training infrastructure where practical;
-   replay recording;
-   evaluation;
-   WebAssembly-compatible environment code.

Suggested crates to evaluate:

-   `serde` / `serde_json` --- configuration and serialisation;
-   `rand` / `rand_chacha` --- deterministic RNG;
-   `ndarray` --- numerical arrays;
-   `rayon` --- parallel CPU simulation;
-   `tokio` --- asynchronous native services if required;
-   `tracing` --- diagnostics;
-   `wasm-bindgen` --- browser/WASM interface;
-   `candle` or `burn` --- neural networks in Rust;
-   `plotters` only if server/native plotting is ever needed.

**Recommendation:** start with tabular RL implemented directly in Rust.
Do not choose the final deep-learning framework until the basic
environment/training API is stable. Prototype both Burn and Candle
before committing.

### Web frontend

Recommended starting point:

-   **Rust + Leptos** for the web application;
-   WebAssembly for browser-side simulation;
-   HTML/CSS for the application shell;
-   Canvas or WebGL for the arena renderer.

Alternative: use a TypeScript frontend with the Rust simulation compiled
to WASM. This may ultimately provide a larger UI ecosystem, but a
Rust/Leptos frontend keeps the initial project predominantly in one
language.

### Training architecture

Support two execution modes eventually:

1.  **Browser mode** --- small tabular experiments and replay.
2.  **Native trainer** --- CPU/GPU training for DQN/PPO/SAC and large
    experiments.

The same environment definitions should be usable by both.

Do **not** require rendering during training.

------------------------------------------------------------------------

## 3. Repository Structure

Start with a Cargo workspace:

``` text
rl-arena/
├── Cargo.toml
├── README.md
├── docs/
│   ├── design.md
│   ├── levels.md
│   └── rl-concepts.md
├── crates/
│   ├── rl-core/
│   │   ├── environment
│   │   ├── observation
│   │   ├── action
│   │   ├── reward
│   │   └── metrics
│   ├── rl-envs/
│   │   ├── gridworld
│   │   └── arena
│   ├── rl-agents/
│   │   ├── q_learning
│   │   └── random
│   ├── rl-training/
│   │   ├── trainer
│   │   ├── evaluator
│   │   └── replay
│   └── rl-web/
│       ├── ui
│       ├── renderer
│       └── wasm
├── apps/
│   └── trainer/
└── tests/
```

Keep `rl-core` independent of graphics, UI and specific ML frameworks.

------------------------------------------------------------------------

## 4. Core Environment API

Define the abstraction before building many levels.

Conceptually:

``` rust
pub trait Environment {
    type Observation;
    type Action;

    fn reset(&mut self, seed: u64) -> Self::Observation;

    fn step(&mut self, action: Self::Action) -> StepResult<Self::Observation>;

    fn valid_actions(&self) -> Vec<Self::Action>;
}
```

A step should provide at least:

``` rust
pub struct StepResult<O> {
    pub observation: O,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub info: StepInfo,
}
```

The API should deliberately resemble Gymnasium's:

``` text
reset()
step(action)
observation
reward
terminated
truncated
info
```

This will make later interoperability much easier.

### Non-negotiable requirements

The simulation must support:

-   deterministic seeds;
-   headless execution;
-   fast reset;
-   fast stepping;
-   serialisable configuration;
-   episode recording;
-   reproducible evaluation;
-   interchangeable agents;
-   interchangeable opponents.

------------------------------------------------------------------------

# 5. Development Milestones

## Milestone 0 --- Project skeleton and technical spike

**Goal:** prove that Rust simulation code can drive a browser
visualisation.

Build:

-   Cargo workspace;
-   `rl-core`;
-   `rl-envs`;
-   `rl-web`;
-   minimal Leptos application;
-   Rust/WASM bridge;
-   5×5 grid rendered in browser;
-   keyboard-controlled agent;
-   seeded environment reset.

No RL yet.

### Exit criteria

You can open the application in a browser, reset a seeded GridWorld,
move an agent around it, and run the same environment headlessly from a
Rust test.

------------------------------------------------------------------------

## Milestone 1 --- Level 1: Reach the Target

**RL topic:** tabular Q-learning.

Environment:

-   5×5 GridWorld;
-   agent position;
-   target position;
-   four discrete actions;
-   deterministic transitions.

State:

``` text
(x, y)
```

Rewards:

``` text
target: +10
step:   -1
```

Implement:

-   random agent;
-   Q-learning agent;
-   epsilon-greedy exploration;
-   learning rate;
-   discount factor;
-   episode runner;
-   Q-table persistence.

### Web UI

Create the first version of the main gameplay workflow:

``` text
CONFIGURE → TRAIN → ANALYSE → DEPLOY
```

Display:

-   arena;
-   current episode;
-   cumulative reward;
-   training progress;
-   Q-values for a selected cell;
-   policy arrows over cells.

### Exit criteria

A player can train an agent from the browser UI and watch the learned
policy reach the target.

------------------------------------------------------------------------

## Milestone 2 --- RL experiment infrastructure

**Goal:** build the infrastructure required by every subsequent level
before adding complexity.

Implement an experiment configuration:

``` text
environment
seed
algorithm
hyperparameters
training budget
evaluation episodes
```

Collect:

-   episode return;
-   episode length;
-   success rate;
-   exploration rate;
-   training steps;
-   wall-clock training time.

Implement:

-   experiment save/load;
-   policy save/load;
-   replay recording;
-   replay playback;
-   deterministic evaluation;
-   comparison between two training runs.

### Exit criteria

Two experiments can be run with different hyperparameters and compared
reproducibly.

------------------------------------------------------------------------

## Milestone 3 --- Level 2: Dangerous Terrain

**RL topic:** reward engineering.

Add:

-   walls;
-   hazards;
-   lethal cells;
-   alternative paths.

Expose reward configuration to the player.

Example:

``` text
goal       +100
death      -100
step         -1
hazard       -5
```

Design scenarios where poorly selected rewards produce undesirable
behaviour.

### Teaching objective

The player should discover that successful RL requires defining the
objective correctly, not merely choosing an algorithm.

### Exit criteria

At least two deliberately bad reward configurations produce recognisably
pathological policies.

------------------------------------------------------------------------

## Milestone 4 --- Level 3: The Hunter

**RL topic:** state representation and adversarial behaviour.

Add a scripted deterministic hunter.

Initial inadequate state:

``` text
agent_position
```

Corrected state:

``` text
agent_position
hunter_position
```

The player should be able to choose which observations are exposed to
the policy.

### UI addition

Create an **Observation Builder**:

``` text
[x] agent position
[ ] hunter position
[ ] hunter direction
[ ] previous action
```

### Teaching objective

This is the first major conceptual transition:

> A policy cannot learn to react to information that is absent from its
> state/observation.

### Exit criteria

The player can demonstrate experimentally that adding hunter information
substantially improves performance.

**This milestone is the first major playable prototype.**

------------------------------------------------------------------------

## Milestone 5 --- Level 4: Stochastic World

**RL topic:** stochastic transitions and MDPs.

Add:

-   probabilistic movement;
-   stochastic hunter behaviour;
-   configurable transition probabilities.

Example:

``` text
requested NORTH

80% NORTH
10% WEST
10% EAST
```

Add visualisation of repeated outcomes from the same state/action pair.

### Exit criteria

The player can compare deterministic and stochastic policies and
understand why a policy must account for expected outcomes.

------------------------------------------------------------------------

## Milestone 6 --- Level 5: Limited Vision

**RL topic:** partial observability.

Restrict observations to:

-   local cells;
-   approximate enemy direction;
-   visible hazards;
-   selected sensor values.

Clearly distinguish:

``` text
true environment state
        ≠
agent observation
```

Provide a developer/debug view showing both.

### Exit criteria

The same underlying world can produce policies with different
performance depending on the observation configuration.

------------------------------------------------------------------------

## Milestone 7 --- Level 6: Memory

**RL topic:** history and recurrent policies.

Create tasks in which the current observation is insufficient.

Examples:

-   switches opening remote doors;
-   temporarily visible targets;
-   enemy sightings that disappear;
-   delayed objectives.

First support:

-   observation/frame stacking;
-   previous action;
-   previous reward.

Then introduce a recurrent policy once deep RL infrastructure exists.

### Exit criteria

A memory-dependent scenario exists in which a memoryless policy
demonstrably underperforms.

------------------------------------------------------------------------

# 6. Deep RL Milestone

## Milestone 8 --- Neural network training infrastructure

Before Level 7, introduce a native training executable.

Requirements:

``` text
environment
    ↓
parallel simulation workers
    ↓
experience collection
    ↓
replay/rollout storage
    ↓
neural network
    ↓
optimiser
    ↓
policy checkpoint
```

Implement:

-   tensor conversion for observations;
-   model checkpoints;
-   CPU training;
-   GPU training where supported;
-   training cancellation;
-   periodic evaluation;
-   policy export;
-   browser loading of completed policies where feasible.

First implement **DQN**.

Do not implement PPO/SAC/MARL simultaneously.

### Exit criteria

DQN solves an environment already solved by tabular Q-learning,
providing a regression test for the deep-RL pipeline.

------------------------------------------------------------------------

## Milestone 9 --- Level 7: Large State Space

**RL topic:** function approximation / DQN.

Increase environment complexity sufficiently that a Q-table becomes
impractical.

Observation may include:

``` text
position
velocity
orientation
enemy position
sensor values
```

Teach the conceptual transition:

``` text
Q[state][action]
```

to:

``` text
Qθ(state, action)
```

### Exit criteria

DQN learns a useful policy and the UI can display training loss, return
and evaluation success.

------------------------------------------------------------------------

## Milestone 10 --- Level 8: Continuous Control

**RL topic:** policy-gradient / actor-critic methods.

Move to a continuous or semi-continuous arena.

Example actions:

``` text
throttle ∈ [-1, 1]
steering ∈ [-1, 1]
```

Implement **PPO** first.

Add SAC later if there is a clear gameplay/teaching reason.

Possible scenario:

-   vehicle;
-   spacecraft;
-   drone;
-   robot.

### Exit criteria

The player trains and deploys a continuous-control policy.

------------------------------------------------------------------------

## Milestone 11 --- Level 9: Resource Management

**RL topic:** delayed consequences and expanded state.

Add:

-   health;
-   energy;
-   ammunition;
-   sensors;
-   objectives;
-   resource pickups.

Example observation:

``` text
position
velocity
orientation
health
energy
ammo
enemy observation
objective state
```

### Teaching objective

A locally attractive action may have a poor long-term return.

### Exit criteria

Successful policies must manage at least two limited resources while
pursuing an objective.

------------------------------------------------------------------------

# 7. Adversarial RL Campaign

## Milestone 12 --- Level 10: Multiple Scripted Opponents

**RL topic:** robustness and training distributions.

Create opponent archetypes:

-   random;
-   aggressive;
-   defensive;
-   patrol;
-   ambush;
-   evasive.

Allow training against:

-   one opponent;
-   random opponent selection;
-   weighted opponent distributions.

### Teaching objective

``` text
training distribution ≠ deployment distribution
```

### Exit criteria

A policy overfitted to one opponent can be defeated by another, while a
robustly trained policy performs better across the opponent set.

------------------------------------------------------------------------

## Milestone 13 --- Level 11: Learning Opponent

**RL topic:** self-play.

Allow both sides to use trainable policies.

Add policy snapshots:

``` text
opponent_v001
opponent_v002
opponent_v003
...
```

Implement training against historical checkpoints.

Track:

-   win rate;
-   draw rate;
-   Elo-style rating if useful;
-   performance against historical policies.

### Exit criteria

An agent improves through self-play without immediately forgetting how
to defeat older strategies.

------------------------------------------------------------------------

## Milestone 14 --- Level 12: Adversarial Environment

**RL topic:** robust RL and domain randomisation.

The adversary can manipulate environment parameters:

-   obstacles;
-   spawn positions;
-   resource locations;
-   sensor noise;
-   path blocking;
-   traps.

Model conceptually:

``` text
π_agent vs π_environment
```

Initially make the environment adversary scripted. A trainable
adversarial environment can follow later.

### Exit criteria

Policies trained under domain randomisation generalise better than
policies trained in one fixed environment.

------------------------------------------------------------------------

# 8. Multi-Agent Campaign

## Milestone 15 --- Level 13: Two versus Two

**RL topic:** multi-agent reinforcement learning.

Support:

``` text
Agent A1 ─┐
          ├── Team A
Agent A2 ─┘

             VS

Agent B1 ─┐
          ├── Team B
Agent B2 ─┘
```

Start with shared policies.

Then investigate:

-   independent learners;
-   parameter sharing;
-   centralised training;
-   decentralised execution.

### Exit criteria

Cooperative behaviour emerges that cannot be achieved effectively by a
single agent.

------------------------------------------------------------------------

## Milestone 16 --- Level 14: Communication

**RL topic:** learned communication.

Give agents:

``` text
physical action
+
small communication action
```

For example:

``` text
message ∈ {0..15}
```

Visualise messages during replay.

Investigate whether communication improves team performance and whether
stable protocols emerge.

### Exit criteria

A task exists in which communication measurably improves team success.

------------------------------------------------------------------------

## Milestone 17 --- Level 15: The Arena

Combine:

-   partial observability;
-   memory;
-   continuous or rich discrete control;
-   resource constraints;
-   stochastic environments;
-   multiple agents;
-   adaptive opponents;
-   communication.

This becomes the final sandbox and tournament environment.

### Exit criteria

Policies from multiple training approaches can compete under a standard
evaluation protocol.

------------------------------------------------------------------------

# 9. RL Debugger

Build the debugger incrementally from Milestone 2 onward rather than
leaving it until the end.

Eventually expose:

-   episode return;
-   success/win rate;
-   episode length;
-   Q-values;
-   TD error;
-   neural-network loss;
-   policy entropy;
-   exploration rate;
-   action distribution;
-   training throughput;
-   inference time;
-   state/observation comparison;
-   replay timeline.

Replay should allow:

``` text
pause
step forward
step backward
inspect observation
inspect chosen action
inspect reward
inspect policy output
```

For an educational game, this is a core gameplay system, not merely
developer tooling.

------------------------------------------------------------------------

# 10. Compute Budget Mechanic

Introduce this only after training infrastructure is stable.

Give each challenge a budget such as:

``` text
Environment steps: 1,000,000
Training time:       20 minutes
Model parameters:    optional limit
```

Score policies using some combination of:

``` text
task performance
sample efficiency
training time
model size
generalisation
```

This prevents brute-force training from becoming the universal answer.

Possible challenge score:

``` text
score =
    performance
    - compute_penalty
    - model_complexity_penalty
```

Keep the exact scoring formula configurable.

------------------------------------------------------------------------

# 11. Opponent Escalation

The campaign should approximately follow:

``` text
Random opponent
      ↓
Deterministic scripted opponent
      ↓
Stochastic opponent
      ↓
Strategy-switching opponent
      ↓
Policy-based opponent
      ↓
Learning opponent
      ↓
Self-play
      ↓
Historical policy population
      ↓
Multi-agent opponents
```

This should be reflected in the code architecture: an opponent is
another policy provider, not special-case game logic.

------------------------------------------------------------------------

# 12. Historical Policy / Final Adversary Mechanic

Persist selected player policies throughout the campaign:

``` text
policy_level_05
policy_level_07
policy_level_09
policy_level_11
...
```

Later train adversaries against this policy history.

The final adversary should therefore exploit weaknesses in strategies
that the player previously developed.

This provides a concrete demonstration of adversarial training and
policy exploitation.

------------------------------------------------------------------------

# 13. Web Application Layout

A useful initial layout:

``` text
+-----------------------------------------------------------+
| Level | Environment | Agent | Training | Experiments      |
+----------------------------+------------------------------+
|                            |                              |
|                            | Observations                 |
|          ARENA             | [x] position                |
|                            | [x] enemy position           |
|                            | [ ] previous action          |
|                            |                              |
+----------------------------+------------------------------+
| Training / Evaluation Graphs                              |
|                                                           |
+-----------------------------------------------------------+
| CONFIGURE       TRAIN       ANALYSE       DEPLOY          |
+-----------------------------------------------------------+
```

Later add dedicated views for:

-   campaign;
-   experiment comparison;
-   replay debugger;
-   policy library;
-   self-play league;
-   tournament arena.

------------------------------------------------------------------------

# 14. Testing Strategy

RL software can appear to work while containing subtle errors. Testing
should therefore begin immediately.

### Unit tests

Test:

-   transitions;
-   rewards;
-   terminal conditions;
-   observation construction;
-   deterministic seeds;
-   action validity.

### Property tests

Examples:

-   agent never leaves valid world bounds;
-   health never exceeds maximum;
-   identical seed + actions produces identical trajectory.

### RL regression tests

Keep very small environments with known solutions.

For example:

``` text
2×2 GridWorld
```

The optimal policy is known, so Q-learning and DQN can be tested against
it.

### Performance benchmarks

Measure:

``` text
environment steps / second
reset time
training throughput
WASM performance
native performance
```

Avoid optimising rendering before simulation throughput has been
measured.

------------------------------------------------------------------------

# 15. Development Order for a Sole Developer

Avoid building the entire architecture up front.

The recommended sequence is:

``` text
Rust environment
      ↓
browser renderer
      ↓
manual control
      ↓
Q-learning
      ↓
training UI
      ↓
metrics
      ↓
replay
      ↓
Level 2
      ↓
Level 3 / Hunter
      ↓
stochasticity
      ↓
partial observability
      ↓
memory
      ↓
native neural trainer
      ↓
DQN
      ↓
PPO
      ↓
adversarial RL
      ↓
self-play
      ↓
multi-agent RL
```

Every milestone should leave the project runnable.

------------------------------------------------------------------------

# 16. First Development Sprint

The immediate objective should **not** be to build an RL game.

It should be to build the smallest vertical slice proving the
architecture.

## Step 1 --- Create workspace

Create:

``` text
rl-core
rl-envs
rl-agents
rl-training
rl-web
```

## Step 2 --- Implement GridWorld

Support:

``` text
width
height
start
goal
walls
seed
```

Implement `reset()` and `step()`.

## Step 3 --- Test headless simulation

Write tests that execute thousands of steps without a renderer.

## Step 4 --- Compile environment to WASM

Expose:

``` text
reset
step
state
```

to the web application.

## Step 5 --- Render GridWorld

Display:

-   grid;
-   player;
-   target;
-   walls.

Allow keyboard movement.

## Step 6 --- Implement Q-learning

Keep it deliberately simple and readable.

## Step 7 --- Add training controls

Initial controls:

``` text
episodes
alpha
gamma
epsilon
seed
```

Button:

``` text
TRAIN
```

## Step 8 --- Visualise learned policy

Show directional arrows and Q-values.

## Step 9 --- Add deployment

Allow:

``` text
RUN LEARNED POLICY
```

and animate an evaluation episode.

## Step 10 --- Save experiment

Persist:

``` text
environment configuration
hyperparameters
Q-table
metrics
seed
```

At this point you have the architectural seed for the complete project.

------------------------------------------------------------------------

# 17. First Major Target

The first meaningful release should contain only:

### Level 1

GridWorld + Q-learning.

### Level 2

Hazards + reward engineering.

### Level 3

Hunter + state/observation expansion.

### Experiment system

Training, metrics, replay and comparison.

### Web interface

Configure, Train, Analyse and Deploy.

Do **not** start DQN before this release works well.

Level 3 is the critical validation point because it tests the game's
central educational mechanic:

> The old agent fails because the world now contains information that
> its existing representation does not model.

If this transition is understandable and enjoyable, continue with the
full roadmap.

------------------------------------------------------------------------

# 18. Second Major Target

Add:

-   stochastic transitions;
-   partial observability;
-   memory;
-   native training;
-   DQN;
-   GPU support.

This produces Levels 4--7 and establishes the deep-RL architecture.

------------------------------------------------------------------------

# 19. Third Major Target

Add:

-   continuous arena;
-   PPO;
-   resource management;
-   scripted opponent families;
-   robustness evaluation.

This produces Levels 8--10.

------------------------------------------------------------------------

# 20. Fourth Major Target

Add:

-   trainable opponents;
-   self-play;
-   policy populations;
-   adversarial environment generation;
-   historical policy evaluation.

This produces Levels 11--12 and establishes the adversarial RL system.

------------------------------------------------------------------------

# 21. Final Major Target

Add:

-   2v2 environments;
-   shared/team policies;
-   communication;
-   league/tournament evaluation;
-   final Arena;
-   historical-policy adversary.

This produces Levels 13--15.

------------------------------------------------------------------------

# 22. Scope Rules

As a sole developer, enforce the following rules.

1.  **Simulation before graphics.**
2.  **One new RL concept per level where possible.**
3.  **No deep RL until tabular gameplay is fun and understandable.**
4.  **No self-play until single-agent training is reliable.**
5.  **No MARL until self-play is reliable.**
6.  **Every environment must run headlessly.**
7.  **Every experiment must be reproducible from a seed and
    configuration.**
8.  **Rendering must never be required for training.**
9.  **Educational visualisation is more important than graphical
    fidelity.**
10. **Do not implement an algorithm unless a gameplay problem motivates
    it.**

------------------------------------------------------------------------

# 23. Immediate Backlog

Start with these tasks, in this order:

-   [ ] Create Cargo workspace.
-   [ ] Create `rl-core`.
-   [ ] Define `Environment`, `Agent`, `Observation`, `Action` and
    `StepResult`.
-   [ ] Implement seeded RNG.
-   [ ] Implement 5×5 GridWorld.
-   [ ] Add environment unit tests.
-   [ ] Add headless episode runner.
-   [ ] Implement random agent.
-   [ ] Implement Q-learning agent.
-   [ ] Add Q-learning regression test.
-   [ ] Create Leptos web application.
-   [ ] Compile environment to WASM.
-   [ ] Render GridWorld.
-   [ ] Add keyboard/manual control.
-   [ ] Connect Q-learning trainer.
-   [ ] Add Train button.
-   [ ] Add reward/episode graph.
-   [ ] Add Q-value visualisation.
-   [ ] Add learned-policy deployment.
-   [ ] Add experiment serialisation.
-   [ ] Add replay recording.
-   [ ] Implement Level 2 hazards.
-   [ ] Implement configurable rewards.
-   [ ] Implement Level 3 scripted hunter.
-   [ ] Implement Observation Builder.
-   [ ] Compare agent-only state with agent+hunter state.
-   [ ] Stop and evaluate whether the core game loop is educational and
    enjoyable.

------------------------------------------------------------------------

# 24. Definition of Success for the Initial Prototype

The initial prototype is successful when a new player can:

1.  open the game in a browser;
2.  understand the objective without reading RL code;
3.  manually play the environment;
4.  configure a Q-learning agent;
5.  train it;
6.  watch its learning curve;
7.  inspect its learned Q-values;
8.  deploy the trained policy;
9.  introduce hazards and observe reward-design effects;
10. encounter the Hunter;
11. see their previous state representation fail;
12. add the Hunter to the observation;
13. retrain;
14. observe the resulting improvement.

At that point, the project has demonstrated its central idea and it is
worth investing in DQN, PPO, self-play and multi-agent RL.
