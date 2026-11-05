# Reinforcement Odyssey: The Road to Ithaca

A web game that teaches reinforcement learning through Odysseus' voyage home.
Start with the development plan, `adversarial_rl_game_development_plan.md`, and the design docs in `docs/`.

## Status

Milestones 0 and 1 are in place:

- Rust workspace with a multi-agent `World` trait and a deterministic RNG.
- Grid engine driven by data files in `content/levels/`, with validation and a compile step.
- Level 0 (Calypso's Loom) and Level 1 (The Raft) as level files, with tutorials in `content/tutorials/`.
- Headless episode runner, random agent, and replay from seed + action log.
- Browser client: amber terminal shell, phosphor arena, style-sheet scene, keyboard control.
  The simulation runs in a Web Worker.

- Campaign flow: a voyage map with locked levels, an objective that opens from the map (click a level or the OBJECTIVE chip; Esc closes it), a completion moment, and progress saved in the browser.
  Level 0 is completed by sailing the shortest route by hand. Level 1 is agent-piloted: you watch a random agent fail against your benchmark.
- ANALYSE view for agent-piloted levels: run the agent over up to 10,000 seeds, see return statistics,
  a per-voyage chart with the running mean, a histogram, the best seed (replayable exactly), and Calypso's questions,
  whose answers come from your own batch.

- Tabular Q-learning with the CONFIGURE, TRAIN, ANALYSE and DEPLOY workflow: learning settings and reward editing
  within each level's bounds, training in chunks inside the worker with a live learning curve and a training budget,
  deployment on fixed test seeds with star ratings, and the learned policy drawn over the arena.
- Level 2, The Lotus-Eaters: rocks, whirlpools and a lotus field. The player breeds a lotus-eater and a raft that
  dives into a whirlpool on purpose, then writes rewards that bring the raft home safely. Each lesson is pinned by
  a test that trains the agent (`crates/rl-training/tests/lotus_eaters.rs`).

- Live Q-learning walkthrough (TRAIN → STEP THROUGH): the pseudocode runs with the raft's real numbers, one line,
  one move or one episode at a time, while the arena shows the same step and the updated Q-value flashes. These are
  real training episodes: stepping and bulk training give the identical raft.
- Q-value inspector: click any cell to read its four Q-values.
- Side panels (the RL loop, Q-values, voyage) fold to a one-line summary. They start folded where they repeat
  what the page already shows (the walkthrough, and the panel views), and a tutorial step that points into one unfolds it.
- The phase buttons appear as the tutorial reaches them: ANALYSE when the tutorial first points at measuring, and
  CONFIGURE, TRAIN and DEPLOY (with the trained raft's controls and the Q-values) when it first points at learning.
  In Level 1, the random raft's steps show none of the learning controls. A completed level or a trained raft shows them all.
- Tutorial predictions and guided experiments: the player commits to a prediction, then runs a pinned experiment
  that answers it. Every experiment states what it must show, and a test runs them all.
- Trained rafts are saved per level in the browser (IndexedDB) and continue training exactly where they stopped.
  A raft trained on an older version of a level file is not loaded.
- Training goes through a `TrainingTarget` interface, with this browser's worker as the only target so far.

Next: the Burn hardware spike, then Milestone 2 (experiment specifications, run comparison, the concept library).

## Layout

| Path | Contents |
|---|---|
| `crates/rl-core` | `World` trait, specs, actions, observations, RNG, render model |
| `crates/rl-envs` | Level file format, validation, Grid engine |
| `crates/rl-agents` | Agents: random and tabular Q-learning |
| `crates/rl-training` | Episode runner, step-by-step training, replay, evaluation, statistics |
| `crates/rl-tutorial` | Tutorial file format |
| `crates/rl-theme` | Visual-style colour tokens |
| `crates/rl-web` | Leptos web client, renderer, simulation worker |
| `content/levels` | Level files (RON) |
| `content/tutorials` | Tutorial files (RON) |

## Requirements

- Rust (via [rustup](https://rustup.rs)) with the WebAssembly target:
  `rustup target add wasm32-unknown-unknown`
- [Trunk](https://trunkrs.dev): `cargo install trunk --locked`

## Run the tests

```sh
cargo test --workspace
```

This checks the simulation, every level file, every tutorial file, and the client's simulation core.

## Run the game

```sh
cd crates/rl-web
trunk serve
```

Then open <http://127.0.0.1:8080>.

Controls:

| Key | Action |
|---|---|
| Arrow keys or W A S D | Steer the raft |
| R | Restart the episode with the same seed |
| N | Next seed |
| Space | Walkthrough: step one line |
| Enter | Walkthrough: step one move |

## Test on other computers

- **GitHub Pages:** every push to `main` is tested, built and published to <https://sandrospina.github.io/ithaca/>
  (`.github/workflows/pages.yml`). One-time setup: Settings → Pages → Source: GitHub Actions.
- **Local network:** `scripts/serve-lan.sh` builds the release client and serves it on port 8081 to other
  computers on the same network.

The status bar shows the build's commit. `docs/testing.md` has the checklist for testers.

## Debug URL parameters

Useful for testing and screenshots:

```text
?level=l01_the_raft   load a level
?seed=3               set the seed
?moves=EESS           apply moves on load (N, E, S, W)
?step=2               open the tutorial at a step (0-based)
?view=stylesheet      show the style-sheet scene
?plain=1              plain mode: no glow or trails
?unlock=all           open every level regardless of progress
?watch=1              start the agent sailing on load (agent-piloted levels)
?view=analyse         open the ANALYSE view
?batch=1000           run a batch of that many voyages on load
?view=configure       open CONFIGURE (also train, walkthrough, analyse, deploy)
?rewards=lure:5       start a new raft with these reward weights
?train=800            train that many episodes on load
?deploy=1             deploy once the training on load finishes
```

Progress is stored in the browser's local storage and trained rafts in IndexedDB. Use RESET PROGRESS in the status
bar to start the campaign again; it also forgets the saved rafts. Applying new settings in CONFIGURE forgets that
level's saved raft. `?rewards=` and `?train=` skip loading the saved raft.
