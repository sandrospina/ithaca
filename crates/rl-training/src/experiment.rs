//! Experiments: a versioned, validated job specification and the run it
//! produces.
//!
//! The browser runs experiments in its worker. The training servers will run
//! the same specification (`docs/training-architecture.md` §8), and check it
//! with the same [`ExperimentConfig::prepare`].

use rl_agents::{GreedyPolicy, QLearner, QLearningConfig};
use rl_core::{SpecHash, World};
use rl_envs::{EditMode, EvalSeeds, GoalDef, GridWorld, LevelError, Pilot, PlayerEdits, SuccessDef, TaskDef};
use serde::{Deserialize, Serialize};

use crate::episode::{EpisodeOutcome, PlayedEpisode, RecordedEpisode, play_recorded};
use crate::stats::BatchEpisode;
use crate::trainer::TabularTrainer;

/// Version of [`ExperimentConfig`]. Bump it on any change that alters what a
/// specification means.
pub const EXPERIMENT_SCHEMA: u32 = 1;
/// Most test episodes one evaluation may run.
pub const MAX_EVAL_EPISODES: u32 = 1_000;
/// Most training episodes in a level without a training budget.
pub const MAX_EXPERIMENT_EPISODES: u32 = 20_000;
/// Test seeds for a level without a training task: ten seeds from 1000.
const DEFAULT_TEST_SEEDS: Evaluation = Evaluation { first_seed: 1000, episodes: 10 };

/// One training run, fully specified: the same specification on the same
/// engine version always gives the same result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentConfig {
    pub schema: u32,
    pub level: LevelRef,
    #[serde(default)]
    pub edits: PlayerEdits,
    /// Seed of the agent's own randomness. Training episode `k` sails on seed `seed + k`.
    pub seed: u64,
    pub algorithm: Algorithm,
    /// Training episodes.
    pub episodes: u32,
    /// Test episodes run after training, without exploration.
    pub evaluation: Evaluation,
}

/// The level file a specification was written for.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LevelRef {
    pub id: String,
    /// [`rl_envs::level_hash`] of the level file.
    pub hash: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Algorithm {
    QLearning(QLearningConfig),
}

impl Algorithm {
    pub fn q_learning(&self) -> QLearningConfig {
        match self {
            Algorithm::QLearning(c) => *c,
        }
    }
}

/// Test episodes on seeds `first_seed..first_seed + episodes`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evaluation {
    pub first_seed: u64,
    pub episodes: u32,
}

impl Evaluation {
    /// The level's test seeds: those of its first training task.
    pub fn of_level(world: &GridWorld) -> Self {
        match world.grid().challenge.as_ref().map(|c| &c.goal) {
            Some(GoalDef::Tasks(tasks)) => tasks
                .iter()
                .find_map(|t| match t {
                    TaskDef::TrainAgent { eval_episodes, eval_seeds: EvalSeeds::Fixed(first), .. } => {
                        Some(Evaluation { first_seed: *first, episodes: *eval_episodes })
                    }
                    _ => None,
                })
                .unwrap_or(DEFAULT_TEST_SEEDS),
            Some(GoalDef::SuccessRate { eval_episodes, eval_seeds: EvalSeeds::Fixed(first), .. }) => {
                Evaluation { first_seed: *first, episodes: *eval_episodes }
            }
            _ => DEFAULT_TEST_SEEDS,
        }
    }
}

/// Inclusive bounds for each Q-learning setting in a level.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct QLearningBounds {
    pub alpha: (f32, f32),
    pub gamma: (f32, f32),
    pub epsilon: (f32, f32),
}

impl QLearningBounds {
    pub fn of(world: &GridWorld) -> Self {
        let h = world.grid().player_editable.hyperparameters;
        Self {
            alpha: h.and_then(|h| h.alpha).unwrap_or((0.01, 1.0)),
            gamma: h.and_then(|h| h.gamma).unwrap_or((0.0, 0.999)),
            epsilon: h.and_then(|h| h.epsilon).unwrap_or((0.0, 1.0)),
        }
    }

    /// `config` with every setting moved into range; a non-number becomes the lower bound.
    pub fn clamp(&self, config: QLearningConfig) -> QLearningConfig {
        let clamp = |v: f32, (lo, hi): (f32, f32)| if v.is_finite() { v.clamp(lo, hi) } else { lo };
        QLearningConfig {
            alpha: clamp(config.alpha, self.alpha),
            gamma: clamp(config.gamma, self.gamma),
            epsilon: clamp(config.epsilon, self.epsilon),
        }
    }

    fn check(&self, config: QLearningConfig, errors: &mut Vec<LevelError>) {
        for (name, value, (lo, hi)) in [
            ("alpha", config.alpha, self.alpha),
            ("gamma", config.gamma, self.gamma),
            ("epsilon", config.epsilon, self.epsilon),
        ] {
            if !(value.is_finite() && (lo..=hi).contains(&value)) {
                errors.push(LevelError::new(format!("algorithm.{name}"), format!("{value} is outside {lo} to {hi}")));
            }
        }
    }
}

/// The number of Q-table rows the world's agent needs, if its observation is discrete.
pub fn tabular_states(world: &GridWorld) -> Option<u32> {
    world.spec().agents[0].observation.state_count().and_then(|n| u32::try_from(n).ok())
}

/// Most Q-table rows a tabular agent may have, so a table always fits in memory.
pub const MAX_TABLE_STATES: u32 = 1 << 20;

impl ExperimentConfig {
    /// A Q-learning specification tested on the level's own test seeds.
    pub fn q_learning(
        level_id: &str,
        level_source: &str,
        edits: PlayerEdits,
        config: QLearningConfig,
        seed: u64,
        episodes: u32,
    ) -> Result<Self, Vec<LevelError>> {
        let world = rl_envs::load_world(level_source).map_err(|e| e.0)?;
        Ok(Self {
            schema: EXPERIMENT_SCHEMA,
            level: LevelRef { id: level_id.to_string(), hash: rl_envs::level_hash(level_source) },
            edits,
            seed,
            algorithm: Algorithm::QLearning(config),
            episodes,
            evaluation: Evaluation::of_level(&world),
        })
    }

    /// Checks the specification against the level file it names and returns
    /// the world to train in, with the edits applied. Reports every problem.
    pub fn prepare(&self, level_source: &str) -> Result<GridWorld, Vec<LevelError>> {
        let mut errors = Vec::new();
        if self.schema != EXPERIMENT_SCHEMA {
            errors.push(LevelError::new("schema", format!("expected {EXPERIMENT_SCHEMA}, found {}", self.schema)));
        }
        if self.level.hash != rl_envs::level_hash(level_source) {
            errors.push(LevelError::new("level.hash", format!("written for another version of \"{}\"", self.level.id)));
        }
        let mut world = rl_envs::load_world(level_source).map_err(|e| e.0)?;
        if world.grid().id != self.level.id {
            errors.push(LevelError::new("level.id", format!("expected \"{}\"", world.grid().id)));
        }
        if world.grid().pilot != Pilot::Agent {
            errors.push(LevelError::new("level", "the player steers in this level: there is no agent to train"));
        }
        let budget = world.grid().budget.as_ref().map_or(MAX_EXPERIMENT_EPISODES, |b| b.episodes);
        if !(1..=budget).contains(&self.episodes) {
            errors.push(LevelError::new("episodes", format!("{} is outside 1 to {budget}", self.episodes)));
        }
        if !(1..=MAX_EVAL_EPISODES).contains(&self.evaluation.episodes) {
            errors.push(LevelError::new(
                "evaluation.episodes",
                format!("{} is outside 1 to {MAX_EVAL_EPISODES}", self.evaluation.episodes),
            ));
        }
        QLearningBounds::of(&world).check(self.algorithm.q_learning(), &mut errors);
        if let Err(e) = world.apply_edits(&self.edits, EditMode::Strict) {
            errors.extend(e);
        }
        match tabular_states(&world) {
            Some(n) if n <= MAX_TABLE_STATES => {}
            Some(n) => errors.push(LevelError::new(
                "edits.observation",
                format!("a Q-table with {n} rows is too large; the limit is {MAX_TABLE_STATES}"),
            )),
            None => errors.push(LevelError::new("edits.observation", "Q-learning needs a discrete observation")),
        }
        if errors.is_empty() { Ok(world) } else { Err(errors) }
    }
}

/// What a trained agent does on the test seeds, with every voyage recorded.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub episodes: Vec<RecordedEpisode>,
    pub reached: u32,
    pub died: u32,
    /// Caught by an opponent.
    #[serde(default)]
    pub caught: u32,
    pub out_of_time: u32,
    /// Arrivals by the shortest route, on maps where it is defined.
    pub shortest: Option<u32>,
    /// Arrivals that never ended a step on a hazard.
    pub safe: u32,
    pub mean_return: f32,
    pub mean_length: f32,
}

/// The best possible return, on maps where the shortest route is the best
/// one: no hazards, lures or other agents.
pub fn shortest_route_return(world: &GridWorld) -> Option<f32> {
    let g = world.grid();
    let plain = !g.hazards.iter().chain(&g.lures).chain(&g.lethal).any(|&b| b) && g.spawns.len() == 1;
    plain.then(|| g.best_return(0).map(|(r, _)| r)).flatten()
}

/// The world tests sail in: `world` with the level's own wind restored, so
/// an agent trained in calmer water is still tested in the real sea.
pub fn test_world(world: &GridWorld) -> GridWorld {
    let mut w = world.clone();
    w.set_slip(w.grid().level_slip);
    w
}

/// Whether a test episode counts as a success.
pub fn succeeded(def: SuccessDef, e: &PlayedEpisode, best_return: Option<f32>) -> bool {
    match def {
        SuccessDef::ReachGoal => e.outcome == EpisodeOutcome::ReachedGoal,
        SuccessDef::ShortestRoute => {
            e.outcome == EpisodeOutcome::ReachedGoal && best_return.is_some_and(|b| e.episode_return >= b - 1e-4)
        }
        SuccessDef::SafeArrival => e.outcome == EpisodeOutcome::ReachedGoal && e.hazard_steps == 0,
        SuccessDef::StuckOnLure => e.outcome == EpisodeOutcome::OutOfTime && e.lure_steps * 2 >= e.length,
        SuccessDef::Dies => e.outcome == EpisodeOutcome::Died,
    }
}

impl EvaluationReport {
    /// Sails the greedy policy of `learner` on every test seed, in the level's
    /// own wind whatever the agent trained in. Deterministic.
    pub fn run(world: &GridWorld, learner: &QLearner, evaluation: Evaluation) -> Self {
        let mut world = test_world(world);
        let world = &mut world;
        let best = shortest_route_return(world);
        let episodes: Vec<RecordedEpisode> = (0..u64::from(evaluation.episodes))
            .map(|k| play_recorded(world, &mut GreedyPolicy(&learner.table), evaluation.first_seed.wrapping_add(k)))
            .collect();
        let count = |def: SuccessDef| episodes.iter().filter(|e| succeeded(def, &e.played, best)).count() as u32;
        let n = episodes.len().max(1) as f32;
        Self {
            reached: count(SuccessDef::ReachGoal),
            died: count(SuccessDef::Dies),
            caught: episodes.iter().filter(|e| e.played.outcome == EpisodeOutcome::Caught).count() as u32,
            out_of_time: episodes.iter().filter(|e| e.played.outcome == EpisodeOutcome::OutOfTime).count() as u32,
            shortest: best.map(|_| count(SuccessDef::ShortestRoute)),
            safe: count(SuccessDef::SafeArrival),
            mean_return: episodes.iter().map(|e| e.played.episode_return).sum::<f32>() / n,
            mean_length: episodes.iter().map(|e| e.played.length as f32).sum::<f32>() / n,
            episodes,
        }
    }

    pub fn tested(&self) -> u32 {
        self.episodes.len() as u32
    }

    /// Share of test episodes that reached the goal.
    pub fn success_rate(&self) -> f32 {
        self.reached as f32 / self.tested().max(1) as f32
    }
}

/// Everything one experiment produced.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperimentRun {
    pub config: ExperimentConfig,
    /// The engine that ran it. Results reproduce exactly on the same version.
    pub engine: String,
    /// Hash of the observation and action specs the policy was trained on.
    pub spec_hash: SpecHash,
    /// Every training episode: return, length, arrival and exploratory moves.
    pub curve: Vec<BatchEpisode>,
    pub evaluation: EvaluationReport,
    /// The trained agent.
    pub policy: QLearner,
}

/// The engine name recorded in every run.
pub fn engine_version() -> String {
    format!("rl-training {}", env!("CARGO_PKG_VERSION"))
}

impl ExperimentRun {
    /// Moves made during training.
    pub fn training_steps(&self) -> u64 {
        self.curve.iter().map(|e| u64::from(e.length)).sum()
    }

    /// Share of training moves chosen at random by exploration.
    pub fn exploration_rate(&self) -> f32 {
        let explored: u64 = self.curve.iter().map(|e| u64::from(e.explored)).sum();
        explored as f32 / self.training_steps().max(1) as f32
    }

    /// A fingerprint of everything the run produced: the curve, the test
    /// voyages and the Q-table. Two runs with the same fingerprint are the same.
    pub fn fingerprint(&self) -> u64 {
        let mut h = Fnv(0xcbf2_9ce4_8422_2325);
        for e in &self.curve {
            h.u64(e.seed);
            h.u32(e.episode_return.to_bits());
            h.u32(e.length);
            h.u32(u32::from(e.reached_goal));
            h.u32(e.explored);
        }
        for e in &self.evaluation.episodes {
            h.u64(e.played.seed);
            e.actions.iter().for_each(|&a| h.u32(a));
        }
        let t = &self.policy.table;
        for s in 0..t.states {
            t.row(s).iter().for_each(|v| h.u32(v.to_bits()));
        }
        h.0
    }
}

struct Fnv(u64);

impl Fnv {
    fn bytes(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 = (self.0 ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    fn u32(&mut self, v: u32) {
        self.bytes(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
}

/// An experiment ready to train: the world with the edits applied and a fresh agent.
#[derive(Clone, Debug)]
pub struct PreparedExperiment {
    pub config: ExperimentConfig,
    pub world: GridWorld,
    pub trainer: TabularTrainer,
}

impl ExperimentConfig {
    /// Validates the specification and sets up the world and a fresh agent.
    pub fn start(&self, level_source: &str) -> Result<PreparedExperiment, Vec<LevelError>> {
        let world = self.prepare(level_source)?;
        let states = tabular_states(&world).expect("checked by prepare");
        let trainer = TabularTrainer::new(states, 4, self.algorithm.q_learning(), self.seed);
        Ok(PreparedExperiment { config: self.clone(), world, trainer })
    }
}

impl PreparedExperiment {
    /// Trains for the specification's episodes, then tests on its evaluation seeds.
    pub fn run(&mut self) -> ExperimentRun {
        let left = self.config.episodes.saturating_sub(self.trainer.episodes_trained());
        self.trainer.train(&mut self.world, left);
        let evaluation = EvaluationReport::run(&self.world, &self.trainer.learner, self.config.evaluation);
        ExperimentRun {
            config: self.config.clone(),
            engine: engine_version(),
            spec_hash: self.world.spec().agents[0].spec_hash(),
            curve: self.trainer.curve.clone(),
            evaluation,
            policy: self.trainer.learner.clone(),
        }
    }
}

/// Validates `config` against `level_source`, trains, and evaluates.
pub fn run_experiment(config: &ExperimentConfig, level_source: &str) -> Result<ExperimentRun, Vec<LevelError>> {
    Ok(config.start(level_source)?.run())
}

#[cfg(test)]
mod tests {
    use super::*;

    const RAFT: &str = include_str!("../../../content/levels/l01_the_raft.ron");
    const LOTUS: &str = include_str!("../../../content/levels/l02_the_lotus_eaters.ron");
    const LOOM: &str = include_str!("../../../content/levels/l00_calypsos_loom.ron");

    fn raft(config: QLearningConfig, episodes: u32) -> ExperimentConfig {
        ExperimentConfig::q_learning("l01_the_raft", RAFT, PlayerEdits::default(), config, 1, episodes).unwrap()
    }

    #[test]
    fn a_specification_round_trips_as_json() {
        let c = raft(QLearningConfig::default(), 300);
        assert_eq!(c.evaluation, Evaluation { first_seed: 1000, episodes: 10 });
        let json = serde_json::to_string(&c).unwrap();
        assert_eq!(serde_json::from_str::<ExperimentConfig>(&json).unwrap(), c);
    }

    #[test]
    fn the_same_specification_gives_the_same_run() {
        let c = raft(QLearningConfig::default(), 300);
        let a = run_experiment(&c, RAFT).unwrap();
        let b = run_experiment(&c, RAFT).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.fingerprint(), b.fingerprint());
        assert_eq!(a.curve.len(), 300);
        assert_eq!(a.evaluation.tested(), 10);
        assert!(a.exploration_rate() > 0.0 && a.exploration_rate() < 0.3, "ε = 0.1: {}", a.exploration_rate());
    }

    #[test]
    fn two_settings_compare_reproducibly() {
        // The Level 1 lesson on γ: a short-sighted raft does worse.
        let far = run_experiment(&raft(QLearningConfig { gamma: 0.95, ..Default::default() }, 300), RAFT).unwrap();
        let near = run_experiment(&raft(QLearningConfig { gamma: 0.1, ..Default::default() }, 300), RAFT).unwrap();
        assert_ne!(far.fingerprint(), near.fingerprint());
        assert_eq!(far.evaluation.shortest, Some(10));
        assert!(near.evaluation.shortest.unwrap() < 10, "{:?}", near.evaluation);
    }

    #[test]
    fn a_recorded_test_voyage_replays_from_its_action_log() {
        let run = run_experiment(&raft(QLearningConfig::default(), 300), RAFT).unwrap();
        let mut world = rl_envs::load_world(RAFT).unwrap();
        for e in &run.evaluation.episodes {
            world.reset(e.played.seed);
            let mut total = 0.0;
            for &a in &e.actions {
                total += world.step(&vec![rl_core::Action::Discrete(a)]).agents[0].reward;
            }
            assert_eq!((total, world.step_count()), (e.played.episode_return, e.played.length));
        }
    }

    #[test]
    fn invalid_specifications_are_refused_with_every_reason() {
        let mut c = raft(QLearningConfig { alpha: 2.0, ..Default::default() }, 501);
        c.edits.rewards = vec![("step".into(), 5.0)];
        c.evaluation.episodes = 0;
        let errors = c.prepare(RAFT).unwrap_err();
        let fields: Vec<&str> = errors.iter().map(|e| e.field.as_str()).collect();
        assert_eq!(fields, ["episodes", "evaluation.episodes", "algorithm.alpha", "edits.rewards[0]"]);
        assert!(errors[0].message.contains("1 to 500"), "the level's budget is the limit: {}", errors[0]);
        assert!(errors[3].message.contains("fixed"), "{}", errors[3]);

        let mut stale = raft(QLearningConfig::default(), 10);
        stale.level.hash ^= 1;
        assert_eq!(stale.prepare(RAFT).unwrap_err()[0].field, "level.hash");
        assert!(raft(QLearningConfig::default(), 10).prepare(LOTUS).is_err(), "another level");
        let manual =
            ExperimentConfig::q_learning("l00_calypsos_loom", LOOM, PlayerEdits::default(), Default::default(), 1, 10)
                .unwrap();
        assert!(manual.prepare(LOOM).is_err(), "Level 0 has no agent to train");
    }

    #[test]
    fn reward_edits_must_stay_in_range() {
        let edits = |w: f32| PlayerEdits { rewards: vec![("lure".into(), w)], ..Default::default() };
        let lotus = |w| {
            ExperimentConfig::q_learning("l02_the_lotus_eaters", LOTUS, edits(w), QLearningConfig::default(), 1, 10)
                .unwrap()
        };
        assert!(lotus(5.0).prepare(LOTUS).is_ok());
        assert!(lotus(500.0).prepare(LOTUS).is_err());
        assert!(lotus(f32::NAN).prepare(LOTUS).is_err());
    }
}
