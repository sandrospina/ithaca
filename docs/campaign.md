# Campaign — The Road to Ithaca

Decision (2026-11-02): the campaign uses the Odyssey as a journey with a final goal.

Title (decided 2026-11-02): **Reinforcement Odyssey: The Road to Ithaca**.

Name check on 2026-11-02, by web search only: no game, app, course or trademark named "Reinforcement Odyssey" was found. Two Odyssey-themed games use "Ithaca" in their titles: *Ithaca* by The Pixel Hunt, and *Ithaca: Road Odyssey*. The subtitle is a common phrase, so this is low risk, but it should not be styled to resemble either game. A formal search of the USPTO and EUIPO trademark registers was not done.

## 1. Frame story

Odysseus is held on Ogygia by **Calypso**, whose name means "she who conceals".
She is the game's mentor and narrator, and she speaks through the amber terminal shell.
She cannot sail him home herself, so she teaches him to build agents that learn the way.

Calypso fits the teaching role because she controls what is hidden.
In the partial-observability and memory levels she literally withholds information, and the debug view "lifts her veil" to show the true state.

Each level is a re-imagined episode of the voyage. Homer's order is followed loosely, and the RL curriculum takes priority when the two conflict.

The final goal is **Ithaca**. The suitors who occupy the palace have been trained on **the player's own historical policies**.
Homer's suitors spent years in Odysseus' house learning his household. In the game they spent the campaign learning the player's strategies, by imitating the player's recorded trajectories (Level 13's technique) and then training against the player's saved policies.

## 2. Level map

| # | Episode | RL topic | Why the myth fits |
|---|---|---|---|
| 0 | **Calypso's Loom** (tutorial) | Manual control, environment loop | Sail a raft by hand. The player learns state, action and reward by being the agent |
| 1 | **The Raft** | Tabular Q-learning | Reach the shore of Ogygia. A simple goal-reaching grid |
| 2 | **The Lotus-Eaters** | Reward engineering | The lotus gives small, repeated rewards. Badly shaped rewards make the agent stay and eat forever, which is reward hacking as myth |
| 3 | **The Cyclops' Cave** | State representation, scripted adversary | Polyphemus hunts you. An agent that cannot observe him fails. Add him to the observation and retrain. *First playable prototype* |
| 4 | **The Bag of Winds** | Stochastic transitions | Aeolus' winds push the ship off course with known probabilities |
| 5 | **The House of Hades** | Partial observability | Darkness and shades. Only nearby cells are visible |
| 6 | **Circe's Instructions** | Memory | Circe gives directions early that are needed much later. A memoryless policy forgets them |
| 7 | **The Wine-Dark Sea** | Function approximation, DQN | Open-sea navigation with too many states for a table |
| 8 | **Scylla and Charybdis** | Continuous control, PPO | Steer through the strait with continuous throttle and rudder |
| 9 | **The Cattle of Helios** | Resource management, delayed consequences | The starving crew may eat the sacred cattle. It is locally attractive and catastrophic later |
| 10 | **The Phaeacian Games** | Robustness, opponent distributions | Compete against athletes with different styles. Overfitting to one loses to the others |
| 11 | **Wrestling Proteus** | Learning opponent, self-play | The shape-shifter adapts to every hold you learn |
| 12 | **Poseidon's Wrath** | Adversarial environment, domain randomisation | The sea god rearranges storms, reefs and currents against you |
| 13 | **The Tale at Alcinous' Hall** | Offline RL, behaviour cloning, decision transformer | Odysseus retells his whole voyage to the Phaeacians. The agent learns only from recorded voyages, never from new exploration. Its target "return-to-go" is literally how much of the way home remains |
| 14 | **Father and Son** | Multi-agent RL, 2v2 | Odysseus and Telemachus against the suitors' scouts |
| 15 | **The Silent Signal** | Learned communication | Coordinate with Telemachus using a 16-glyph message channel |
| 16 | **Ithaca** | Everything: the final Arena | The suitors are decision transformers trained on your own recorded voyages, then sharpened against your saved policies. They play like you, and exploit your habits. Win the hall |

## 3. Algorithms by level

Each algorithm enters only when a level's problem motivates it (scope rule 10).

| Level | New algorithm or technique | What fails without it |
|---|---|---|
| 0 | Human player, random agent | — |
| 1 | Tabular Q-learning, ε-greedy | Random play rarely reaches the shore |
| 2 | Same algorithm, configurable rewards | Bad rewards, not a bad algorithm |
| 3 | Same, plus the Observation Builder | Hunter is invisible to the table |
| 4 | Same, under stochastic transitions | Deterministic plans break in the wind |
| 5 | Same, under partial observations | Different true states look identical |
| 6 | Frame stacking, previous action and reward in the observation | Memoryless table forgets Circe's directions |
| 7 | DQN, with replay buffer and target network | The table is too large to fill |
| 8 | PPO | DQN cannot output continuous throttle and rudder |
| 9 | PPO with a recurrent policy (LSTM/GRU) | Long horizons and hidden resource state. This also revisits Level 6 with learned memory |
| 10 | PPO trained on an opponent mixture | A policy overfits to one opponent |
| 11 | Self-play with a league of past snapshots | A fixed training opponent no longer represents the adversary |
| 12 | Domain randomisation, scripted adversarial environment | A policy overfits to one sea |
| 13 | Behaviour cloning, then **decision transformer** | Exploration is forbidden: only recorded voyages are available |
| 14 | Shared-parameter PPO, then centralised-critic PPO (MAPPO) | Independent learners fail to coordinate |
| 15 | MAPPO with a discrete message action | Teammates lack each other's observations |
| 16 | Decision-transformer suitors fine-tuned by self-play against the player's policy library | — |

SAC is not planned. It is added only if a continuous-control level shows a clear need.

### Level 13 in detail: The Tale at Alcinous' Hall

**Dataset.** A dataset must come from one environment, because observations and actions differ between levels. Level 13 therefore revisits the **Wine-Dark Sea** world of Level 7 and uses trajectories the player already owns there:
- every evaluation replay from the player's Level 7 training runs, including early checkpoints that sailed badly;
- any voyages the player sailed manually in that world;
- new manual voyages the player can record at the start of Level 13.

Mixed-quality data is the point: it is exactly what a decision transformer is good at using.

Replays are stored as seed plus action log, so observations and rewards are rebuilt by re-running the deterministic simulation. The dataset costs almost no storage.

**Design constraint for Level 16.** The Ithaca suitors need trajectories recorded in the Ithaca arena. They come from the player's own matches in Levels 14–15, which share Ithaca's arena format, and from compatible saved policies replayed there. Levels 14–16 must therefore share one observation and action format.

**Step 1, behaviour cloning.** The agent copies every recorded voyage. It inherits the player's mistakes along with their successes. This failure motivates step 2.

**Step 2, decision transformer.** The model is conditioned on a target return. The UI shows this as a **"Return-to-go" dial**: the remaining distance to Ithaca.
- Dial set high: the agent imitates the best voyages in the dataset.
- Dial set low: it deliberately sails badly.
- Dial set above anything in the data: it fails. This teaches that offline RL cannot exceed the quality and coverage of its data.

**Comparison.** The player trains PPO on the same task with the same budget and compares the two on the level's challenge. PPO needs interaction with the world. The decision transformer needs only data.

**Why it belongs before Ithaca.** The finale's suitors are built with this exact technique, so the player understands how the final adversary was made.

**Data consent.** By default a player's decision transformer trains only on their own recordings. A shared pool of other players' voyages is opt-in. A teacher can enable it for a class.

## 4. Narrative rules

0. **Every level has a full tutorial.** Its structure, depths and content are defined in `tutorials.md`.
1. **Story never hides the lesson.** Every level opens with a one-screen briefing in Calypso's voice that states the challenge in plain language and the RL concept in one sentence.
2. **Predict before training.** Calypso asks a question before each run, for example "Will this crew ever leave the lotus field?" The answer is revealed by the experiment.
3. **The old agent fails on screen.** At each transition, the player's previous policy is deployed first and visibly fails. That failure motivates the new concept.
4. **Policies are the player's crew.** Saved policies appear in the policy library as named crew members, e.g. "Eurylochus, trained at the Cyclops' cave". This makes the historical-policy mechanic of Ithaca personal.
5. **Map screen.** The campaign view is a vector-drawn Aegean chart with the route lit in phosphor as levels are completed.

## 5. Level challenge format

Every level defines a measurable win condition for the player, separate from the developer exit criterion in the development plan.

```text
level:            3 — The Cyclops' Cave
challenge:        escape in ≥ 8 of 10 seeded evaluation episodes
training budget:  2,000 episodes
stars:            ★ pass   ★★ ≤ 1,000 episodes   ★★★ ≤ 500 episodes
```

The budget applies from Level 1, so brute-force training is never the universal answer. The full compute-budget mechanic from the development plan extends this later.
