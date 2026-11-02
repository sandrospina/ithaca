# Environment Model — Data-Driven Levels on Coded Mechanics

Decision (2026-11-02): levels are **data**, mechanics are **code**.

- A level file *composes* a world from a fixed library of entities, components, behaviours, reward terms and observation features, and sets their parameters.
- Every mechanic, i.e. how something moves, collides, damages, senses or rewards, is implemented once in Rust, tested, and reused.
- There is **no scripting language** inside level files.

## 1. Why this split

| Concern | Hard-coded levels | Fully scripted levels (Lua, Rhai…) | **Data + coded mechanics** |
|---|---|---|---|
| Add a level without code | no | yes | yes, if the mechanics exist |
| Runs safely on shared GPU workers | yes | risky: player/teacher code on your machines | yes: data only, validated |
| Deterministic across native and WASM | yes | depends on the interpreter | yes |
| Fast enough for millions of steps | yes | slower | yes, after compilation (section 6) |
| Observation Builder and reward editor are just data edits | no | partly | yes |
| Teachers can author class scenarios | no | yes, unsafely | yes |
| Testable | per level | hard | every mechanic once + every level file validated in CI |

This matches the rules already set for job configs (`training-architecture.md` §7) and tutorials (`tutorials.md` §6): **data in a fixed vocabulary, never code**.

## 2. World engines

A small number of engines, each multi-agent capable from day one, so Levels 14–16 never force a rewrite.

| Engine | Space | Time | Used by levels | Determinism |
|---|---|---|---|---|
| **Grid** | Discrete cells, 4 or 8 moves | Discrete steps | 0–6 | Integer state |
| **Sea** | Continuous 2D, rigid bodies, headings, currents | Fixed timestep, e.g. 20 Hz | 7–16 | Fixed-point arithmetic (`fixed` crate), no `f32` in simulation state |

Both engines implement one multi-agent interface. A single-agent level is simply a world with one learning agent.

```rust
pub trait World {
    fn reset(&mut self, seed: u64) -> Observations;       // one per agent
    fn step(&mut self, actions: &JointAction) -> StepResults;
    fn action_mask(&self, agent: AgentId) -> ActionMask;   // fixed size, no allocation
    fn spec(&self) -> &WorldSpec;                          // observation + action shapes
}
```

Scripted opponents and trained opponent policies are both `PolicyProvider`s attached to non-learning agents. The world never contains special-case opponent logic (plan §11).

The Gymnasium-style single-agent API from the original plan becomes a thin wrapper over this interface.

## 3. Building blocks

### Entities and components

An entity is an id plus a set of components. Entity counts are small, so a simple struct-of-arrays store is enough. No general ECS framework is needed.

| Component | Parameters | First used |
|---|---|---|
| `Body` | position, heading, velocity, radius / cell | 0 |
| `Agent` | team, action space, controller (learner / scripted / trained policy id) | 0 |
| `Goal` | reward term id, consumed or permanent | 1 |
| `Wall` / `Reef` | cell set or polygon | 2 |
| `Hazard` | damage per step, lethal flag | 2 |
| `Lure` | reward per step while inside, e.g. the lotus | 2 |
| `Hunter` | marker for scripted adversary behaviour | 3 |
| `Wind` | transition noise table or current vector field | 4 |
| `Darkness` | visibility mask region | 5 |
| `Switch` / `Door` / `Beacon` | linked entity ids, visibility timer | 6 |
| `Resources` | health, energy, ammo, food: max, start, regen, drain | 9 |
| `Pickup` | resource, amount, respawn time | 9 |
| `Sensor` | range, field of view, noise, ray count | 5 (grid), 7 (sea) |
| `Comms` | vocabulary size, broadcast range | 15 |
| `Glyph` | visual-style token, e.g. `agent`, `adversary`, `hazard` | 0 |

### Scripted behaviours

Coded, parameterised behaviours selected by id:

`random`, `chase_shortest_path`, `chase_greedy`, `patrol(waypoints)`, `ambush(zone, trigger_radius)`, `evade`, `guard(region)`, `stochastic(behaviour, p_random)`, `switching(behaviours, schedule)`.

Level 10's opponent archetypes and Level 12's scripted Poseidon are combinations of these.

### Reward terms

A level's reward is a **weighted sum of named terms** from a fixed catalogue. The reward editor in Level 2 edits the weights.

`goal_reached`, `step`, `death`, `hazard_entered`, `hazard_per_step`, `lure_per_step`, `distance_to_goal_delta`, `resource_gained`, `resource_spent`, `damage_dealt`, `damage_taken`, `opponent_caught`, `caught_by_opponent`, `team_win`, `team_loss`.

### Observation features

An observation is an **ordered list of named feature extractors**. The Observation Builder in Level 3 edits this list.

| Feature | Parameters | Output |
|---|---|---|
| `self_position` | normalised or one-hot | 2 or W·H |
| `self_heading_velocity` | — | 3 |
| `entity_position` | entity id or tag, absolute or relative | 2 per entity |
| `entity_direction_coarse` | entity tag, number of sectors | one-hot sectors |
| `local_view` | radius, channels (walls, hazards, entities) | (2r+1)²·C |
| `ray_sensors` | count, range, noise | rays·channels |
| `resources` | which resources | 1 per resource |
| `previous_action` / `previous_reward` | — | A / 1 |
| `frame_stack` | k, wraps all features above it | k × inner size |
| `messages_received` | — | vocabulary size per teammate |

Each feature declares its size, so the total observation shape is known **before** training.

**Tabular levels**: the feature list is turned into a discrete state index (product of feature cardinalities). The UI shows the resulting Q-table size, which is itself a teaching moment: adding the Cyclops' position multiplies the table by 25.

## 4. Level file

Level files are written in **RON** (readable, supports Rust enums). The coordinator and browser exchange the same structures as JSON. Both are produced by `serde` from one set of Rust types, with a schema version.

Example: Level 3, The Cyclops' Cave.

```ron
Level(
  schema: 1,
  id: "l03_cyclops_cave",
  engine: Grid(width: 7, height: 7, moves: Four),
  map: [
    "#######",
    "#S....#",
    "#.##..#",
    "#..C..#",
    "#.##..#",
    "#....G#",
    "#######",
  ],
  legend: { 'S': Spawn(agent: "odysseus"), 'C': Spawn(agent: "polyphemus"), 'G': Goal(term: "goal_reached") },
  agents: {
    "odysseus":   (controller: Learner, glyph: "agent"),
    "polyphemus": (controller: Scripted(behaviour: chase_shortest_path(every_n_steps: 2)), glyph: "adversary"),
  },
  reward: { "goal_reached": 10.0, "step": -0.1, "caught_by_opponent": -10.0 },
  observation: [ self_position(encoding: Index) ],          // the deliberately inadequate default
  termination: [ GoalReached, CaughtByOpponent ],
  truncation: (max_steps: 100),
  player_editable: (
    observation: (allowed: [ self_position, entity_position(tag: "polyphemus"), entity_direction_coarse(tag: "polyphemus", sectors: 4), previous_action ]),
    reward: (terms: [ "step", "caught_by_opponent" ], range: (-20.0, 20.0)),
    hyperparameters: (alpha: (0.01, 1.0), gamma: (0.0, 0.999), epsilon: (0.0, 1.0)),
  ),
  challenge: (metric: SuccessRate, threshold: 0.8, eval_episodes: 10, eval_seeds: Fixed(1000)),
  budget: (episodes: 2000, stars: [2000, 1000, 500]),
  tutorial: "l03_cyclops_cave",
)
```

Key points:
- **`player_editable`** states exactly what a player may change, and within which bounds. Everything else is locked. The coordinator enforces the same rules on jobs.
- **Maps** can be ASCII for grids, polygons for the sea, or a **generator** with parameters (section 5).
- **Glyphs** reference visual-style tokens only, so new levels cannot break visual consistency.

## 5. Randomisation and generators

- A generator is coded, e.g. `maze(density)`, `archipelago(islands, reef_density)`, `strait(width, whirlpool_strength)`.
- The level file picks a generator and gives each parameter a **fixed value or a distribution**.
- Every parameter drawn from a distribution is drawn from the episode seed, so randomised levels stay reproducible.
- This is how Level 12's domain randomisation and Poseidon's adversarial seas work. The scripted Poseidon chooses generator parameters to minimise the agent's return, inside the same bounds.

## 6. Compile, don't interpret

Data-driven must not mean slow.

1. **Load**: parse the RON/JSON into `LevelDef`.
2. **Validate**: ids exist, spawns in bounds, map rectangular, observation features compatible with the engine, editable bounds valid, budget positive. Errors name the line and field.
3. **Compile**: build a `CompiledWorld` with flat arrays, enums instead of strings, precomputed index tables and reward weights as a fixed vector.
4. **Run**: `reset` and `step` touch only the compiled form. No string lookups, hash maps or dynamic dispatch in the hot loop, except one call per scripted agent.

## 7. Policies are bound to specs

- A trained policy stores the **hash of its observation spec and action spec**.
- Loading a policy into a world with a different spec is refused with a clear message, e.g. "This crew member was trained without seeing the Cyclops."
- Offline datasets (Level 13) and the Ithaca suitors (Level 16) use the same check. This enforces the "one dataset, one environment format" constraint from `campaign.md`.

## 8. How each level is modelled

| # | Level | Engine | Entities and components | Dynamics | Default observation | Player edits |
|---|---|---|---|---|---|---|
| 0 | Calypso's Loom | Grid 5×5 | raft, goal | deterministic | self_position | — (manual control) |
| 1 | The Raft | Grid 5×5 | raft, goal | deterministic | self_position | hyperparameters |
| 2 | The Lotus-Eaters | Grid 8×8 | + walls, hazards, lethal cells, lotus lure | deterministic | self_position | reward weights |
| 3 | The Cyclops' Cave | Grid 7×7 | + Cyclops (`chase_shortest_path`) | deterministic | self_position | observation list |
| 4 | The Bag of Winds | Grid 8×8 | + wind regions, stochastic Cyclops | transition noise table, e.g. 80/10/10 | self + Cyclops position | noise level (in guided experiment) |
| 5 | The House of Hades | Grid 10×10 | + darkness, shades (`patrol`) | stochastic | local_view(r=1) + coarse shade direction | sensor choice and radius |
| 6 | Circe's Instructions | Grid 10×10 | + beacons visible only at start, switches, doors | stochastic | local_view + beacon_if_visible | frame_stack k, previous action/reward |
| 7 | The Wine-Dark Sea | Sea | ship, islands, reefs, harbour; discrete steering actions | currents | heading, velocity, ray_sensors, harbour bearing | network size, DQN hyperparameters |
| 8 | Scylla and Charybdis | Sea | + Scylla (strike zone), Charybdis (vortex field); continuous throttle and rudder | currents, vortex | as Level 7 + vortex bearing | PPO hyperparameters |
| 9 | The Cattle of Helios | Sea + island | + crew resources (food, health), pickups, sacred cattle (`Lure` with delayed penalty) | resource drain | + resources | recurrence on/off, γ |
| 10 | The Phaeacian Games | Sea arena | + one opponent, archetype drawn from a weighted list | — | + opponent rays | opponent mixture weights |
| 11 | Wrestling Proteus | Sea arena | opponent = trained policy snapshots | — | as Level 10 | league settings |
| 12 | Poseidon's Wrath | Sea, generated | generator `archipelago` with parameter distributions, scripted Poseidon | randomised | as Level 10 | randomisation ranges |
| 13 | The Tale at Alcinous' Hall | Sea (Level 7 world) | as Level 7 | as Level 7 | as Level 7 | dataset selection, return-to-go target |
| 14 | Father and Son | Sea arena, 2v2 | 2 teams, objective zone | — | **Ithaca arena spec** | team training mode |
| 15 | The Silent Signal | Sea arena, 2v2 | + Comms (16 symbols), limited sensor range | — | Ithaca arena spec + messages | channel on/off, vocabulary |
| 16 | Ithaca | Sea arena, 2v2 | suitors = decision-transformer policies | — | Ithaca arena spec + messages | everything allowed so far |

Levels 14–16 share one **Ithaca arena spec**, as required for the Level 16 dataset.

## 9. Tooling and testing

- **CI validation**: every file under `content/levels/` is loaded, validated and compiled on every build.
- **Property tests per level**: random joint actions for many seeds. Agents stay in bounds, resources stay within limits, identical seed and actions give identical trajectories.
- **Hot reload** in the dev build: editing a level file updates the running game.
- **Level editor, later**: an in-browser editor that writes the same files. Useful for you first, then for teachers creating class scenarios. Teacher-authored levels are validated by the coordinator like any job config.

## 10. When to build it

| Milestone | Work |
|---|---|
| 0 | `World` trait (multi-agent shape), Grid engine, `LevelDef` with map + spawn + goal, RON loading, validation, compile step. Level 0 is a data file from day one. |
| 1 | Reward terms and observation features as catalogues, even if only two of each exist. |
| 3 (Level 2) | Hazards, lures, reward editor writes `reward` weights. |
| 4 (Level 3) | Scripted behaviours as `PolicyProvider`s. Observation Builder writes `observation`. Spec hashes on policies. |
| 5–7 | Wind, darkness, sensors, beacons, frame stacking. |
| Before 8 | Sea engine in fixed point. Re-express one grid level in the Sea engine as a regression test. |
| 14 (Level 12) | Generators with parameter distributions. |
| Later | In-browser level editor. |
