# Tutorials — Teaching Each Level's Algorithm and Concept

Decision (2026-11-02): every level has a tutorial that explains the RL concept and the algorithm being taught. Tutorials are a core system, built with Level 1, not added at the end.

## 1. The lesson arc

Every level follows the same seven beats, so players always know where they are.

| Beat | Purpose | Example (Level 1, The Raft) |
|---|---|---|
| 1. **Briefing** | Story + the concept in one plain sentence | Calypso: "Your raft must find the shore. Today an agent learns which moves lead home." |
| 2. **The old way fails** | Deploy the previous agent or a random agent and watch it fail | A random agent drifts for hundreds of steps |
| 3. **Explainer** | Interactive walkthrough of the concept and algorithm on a tiny example | One Q-value update, step by step, on a 2×2 grid |
| 4. **Predict** | Player commits to a prediction before training | "If γ = 0.1, will the agent reach the shore?" |
| 5. **Guided experiment** | Scripted runs that reliably show the lesson | Train with γ = 0.1, then γ = 0.95, compare policy arrows |
| 6. **Challenge** | The level's scored task, with a training budget | Reach the shore in ≥ 9 of 10 evaluation episodes |
| 7. **Debrief** | What happened and why, linked to the concept library | "A low γ makes distant rewards nearly worthless…" |

Players can skip beats 3–5 if they want, but the debrief always shows and the concept library entry always unlocks.

## 2. Three depths of explanation

Every explainer is written at three depths. The player chooses with a toggle that is remembered per player.

| Depth | Content | Audience |
|---|---|---|
| **Story** | Metaphor and plain language. No symbols | Friends, newcomers |
| **Intuition** | Diagrams, the update shown with real numbers from the current run | Default |
| **Math** | Equations, pseudocode, links to papers and textbook chapters | Students, a class |

Equations are rendered with KaTeX (or pre-rendered to SVG) in the terminal style.

## 3. Live algorithm walkthroughs

The key teaching tool: **the algorithm's pseudocode, executing**.

- The pseudocode is shown in the amber terminal. As the agent takes a step, the current line highlights.
- Variables in the pseudocode show live values: s, a, r, s′, the TD target, the TD error, the new Q-value.
- The arena shows the same step at the same moment. The changed Q-cell flashes.
- Controls: step one line, step one transition, run one episode, run fast.

The walkthrough runs the **real Rust agent** in a "lesson mode" that emits trace events. It is never a canned animation, so what is taught is exactly what trains.

Example for Level 1:

```text
  Q-LEARNING                                   s=(1,0)  a=EAST
  ─────────────────────────────────────────────────────────────
  for each step of episode:
    choose a from s using ε-greedy(Q)          ε=0.20 → greedy
    take a, observe r, s′                      r=-1  s′=(2,0)
 ▶  target ← r + γ · max_a′ Q(s′, a′)          -1 + 0.9·4.2 = 2.78
    Q(s,a) ← Q(s,a) + α · (target − Q(s,a))    0.0 + 0.5·2.78 = 1.39
    s ← s′
```

Deep RL levels use the same idea at a coarser grain: one transition flowing into the replay buffer, one minibatch, one gradient step, one target-network sync.

## 4. Concept library

A reference section opened from the shell at any time.
- One entry per concept, unlocked when its level's tutorial is reached.
- Each entry has the three depths, a glossary, the level where it was taught, and "replay this explainer".
- Each entry lists further reading: the matching Sutton & Barto chapter or the original paper.
- Doubles as the project's `docs/rl-concepts.md` source, so documentation and game text are one source.

## 5. Per-level tutorial content

| # | Level | Concept taught | Key "aha" experiment | Further reading |
|---|---|---|---|---|
| 0 | Calypso's Loom | Agent, environment, state, action, reward, episode, return | The player is the agent. The HUD labels every piece of the loop as they sail | Sutton & Barto ch. 1, 3 |
| 1 | The Raft | Value of an action, Q-table, TD update, ε-greedy, α, γ | γ = 0.1 vs 0.95. ε = 0 gets stuck, ε = 0.2 finds the shore | S&B ch. 6.5 |
| 2 | The Lotus-Eaters | Reward design, reward hacking, shaping | A small lotus reward makes the agent never leave. Removing the step penalty makes it wander | S&B ch. 3.2; Ng et al. 1999 (shaping) |
| 3 | The Cyclops' Cave | State vs observation, the Markov property | Same algorithm, with and without the Cyclops in the observation | S&B ch. 3.1, 17.3 |
| 4 | The Bag of Winds | Stochastic transitions, expected return, MDPs | Repeat one action from one cell 100 times and plot outcomes. The safe path beats the short path | S&B ch. 3.1 |
| 5 | The House of Hades | Partial observability, state aliasing | Debug view shows two true states the agent sees as identical | Kaelbling et al. 1998 (POMDPs) |
| 6 | Circe's Instructions | Memory, history as state, frame stacking | Stack size 1 vs 4 vs 8 on a delayed-instruction task | S&B ch. 17.3 |
| 7 | The Wine-Dark Sea | Function approximation, generalisation, DQN, replay buffer, target network | Q-table memory counter explodes. DQN generalises to unseen cells | S&B ch. 9; Mnih et al. 2015 |
| 8 | Scylla and Charybdis | Policy gradients, actor-critic, advantage, PPO clipping | Show the action distribution narrowing as the policy learns. Clip off vs on | S&B ch. 13; Schulman et al. 2017 |
| 9 | The Cattle of Helios | Delayed consequences, credit assignment, recurrent policies | Short-horizon γ eats the cattle. Long-horizon γ survives | S&B ch. 12 |
| 10 | The Phaeacian Games | Distribution shift, robustness | Win-rate matrix: trained-against vs evaluated-against | Tobin et al. 2017 |
| 11 | Wrestling Proteus | Self-play, non-stationarity, forgetting, leagues | Latest-only self-play forgets old tricks. A league remembers | Silver et al. 2017; Vinyals et al. 2019 |
| 12 | Poseidon's Wrath | Domain randomisation, adversarial environments | Fixed-sea policy vs randomised-sea policy on unseen seas | Tobin et al. 2017; Pinto et al. 2017 |
| 13 | The Tale at Alcinous' Hall | Offline RL, behaviour cloning, return conditioning, transformers | Behaviour cloning copies mistakes. The return-to-go dial controls quality | Chen et al. 2021 (Decision Transformer) |
| 14 | Father and Son | Multi-agent RL, non-stationary teammates, centralised training | Independent learners vs shared critic | Yu et al. 2022 (MAPPO) |
| 15 | The Silent Signal | Emergent communication | Message channel on vs muted. Inspect what each glyph comes to mean | Foerster et al. 2016 |
| 16 | Ithaca | Exploitation of a known policy, adversarial training | The suitors reveal which of the player's habits they exploit | — |

## 6. Tutorials are data, not code

- Each tutorial is a data file (Markdown for text + a RON/JSON step script) under `content/tutorials/<level>/`.
- Steps can: show text at a depth, highlight a UI element, load a seeded scenario, set hyperparameters, run training for N episodes, ask a prediction or quiz question, and unlock a library entry.
- No arbitrary code. The step vocabulary is a fixed enum, so tutorials stay safe and testable.
- Writing tutorial text does not require recompiling the game. Translation can be added later.

## 7. Guided experiments must always work

A guided experiment that fails to show its lesson is worse than none.

- Every guided experiment pins its seed and configuration.
- CI runs every tutorial script headlessly and asserts the intended outcome. For example, "with γ = 0.1, success rate < 20%; with γ = 0.95, success rate > 90%".
- If a code change breaks a lesson, the build fails.

## 8. Classroom support

Because classes are a target audience:
- **Teacher role** in the coordinator: create a class, issue an invite code, lock or unlock levels, see per-student progress.
- **Short quizzes** at the debrief, 2–3 questions. Results are visible to the teacher, never used to block progress.
- **Printable lesson sheets** generated from the same tutorial files: the concept, the experiment, and questions for discussion.
- **Prediction log**: a teacher can see which predictions a class got wrong most often, which shows what to discuss in the next lecture.

## 9. When to build it

| Milestone | Tutorial work |
|---|---|
| 0 | Level 0 tutorial with the HUD labelling of the RL loop. Text-only beats. |
| 1 | Tutorial engine: step script format, the seven beats, depth toggle, live Q-learning walkthrough. |
| 2 | Headless tutorial tests in CI. Concept library. |
| 3–7 | One tutorial per level, written with the level. A level is not done until its tutorial passes CI. |
| 7.5b | Teacher role, quizzes, progress view. |
| 8+ | Deep-RL walkthrough grain: buffer, minibatch, gradient step, network sync. |
