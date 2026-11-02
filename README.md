# Reinforcement Odyssey: The Road to Ithaca

A web game that teaches reinforcement learning through Odysseus' voyage home.
Start with the development plan, `adversarial_rl_game_development_plan.md`, and the design docs in `docs/`.

## Status

Milestone 0 is in place:

- Rust workspace with a multi-agent `World` trait and a deterministic RNG.
- Grid engine driven by data files in `content/levels/`, with validation and a compile step.
- Level 0 (Calypso's Loom) and Level 1 (The Raft) as level files, with tutorials in `content/tutorials/`.
- Headless episode runner, random agent, and replay from seed + action log.
- Browser client: amber terminal shell, phosphor arena, style-sheet scene, keyboard control.
  The simulation runs in a Web Worker.

- Campaign flow: a voyage map with locked levels, an objective strip, a completion moment, and progress saved in the browser.
  Level 0 is completed by sailing the shortest route by hand. Level 1 is agent-piloted: you watch a random agent fail against your benchmark.

Q-learning and the Configure / Train / Analyse / Deploy workflow arrive in Milestone 1.

## Layout

| Path | Contents |
|---|---|
| `crates/rl-core` | `World` trait, specs, actions, observations, RNG, render model |
| `crates/rl-envs` | Level file format, validation, Grid engine |
| `crates/rl-agents` | Agents (random so far) |
| `crates/rl-training` | Episode runner, replay, evaluation |
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
```

Progress is stored in the browser's local storage. Use RESET PROGRESS in the status bar to start the campaign again.
