//! The simulation side of the client: owns the world, the episode and the
//! trained agent, and answers requests. Plain Rust, tested natively.

use rl_agents::{GreedyPolicy, QLearner, QLearningConfig, RandomAgent};
use rl_core::{ActionSpace, PolicyProvider, Rng, StepEvent, World};
use rl_envs::level::{Encoding, FeatureDef};
use rl_envs::{EditMode, EvalSeeds, GoalDef, GridWorld, MAX_BATCH_EPISODES, Pilot, PlayerEdits, TaskDef};
use rl_training::{
    BatchEpisode, BatchReport, ExperimentConfig, PlayedEpisode, QLearningBounds, TabularTrainer, TrainingEpisode,
    play_episode, tabular_states,
};

use crate::content;
use crate::protocol::{
    Baseline, BestRoute, CellPolicy, DeployReport, ExperimentResult, FeatureOption, Landing, LessonMode, LessonMove,
    LessonState, MoveOutcomes, Objective, ObservationStatus, Outcome, OutcomeReport, RewardSetting, SAVE_SCHEMA,
    SailedBy, SavedRaft, SimRequest, SimResponse, Snapshot, TaskResult, TrainingStatus, WindStatus,
};

pub use rl_envs::level_hash;
pub use rl_training::{agent_rng, succeeded};

/// Episodes used to measure the random agent's baseline on a level.
pub const BASELINE_EPISODES: u32 = 100;

/// Largest training chunk per request, so the UI stays responsive.
pub const MAX_TRAIN_CHUNK: u32 = 500;

/// Plays one full episode with the random agent on `seed`.
pub fn random_episode(world: &mut GridWorld, seed: u64) -> BatchEpisode {
    play_episode(world, &mut RandomAgent, seed, &mut agent_rng(seed)).to_batch()
}

/// The specification of a tutorial's guided experiment on `level_id`.
pub fn experiment_spec(level_id: &str, def: &rl_tutorial::ExperimentDef) -> Result<ExperimentConfig, String> {
    let entry = content::level(level_id).ok_or_else(|| format!("unknown level \"{level_id}\""))?;
    let d = QLearningConfig::default();
    let config = QLearningConfig {
        alpha: def.alpha.unwrap_or(d.alpha),
        gamma: def.gamma.unwrap_or(d.gamma),
        epsilon: def.epsilon.unwrap_or(d.epsilon),
    };
    let edits = PlayerEdits {
        rewards: def.rewards.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        observation: def.observation.clone(),
        wind: def.wind,
    };
    ExperimentConfig::q_learning(level_id, entry.level_source, edits, config, def.training_seed, def.episodes)
        .map_err(|e| errors_text(&e))
}

use crate::experiments::display_name;

/// What the raft observes, in words: "raft (1, 1) · Polyphemus to the east".
pub fn describe_state(world: &GridWorld) -> String {
    let g = world.grid();
    let obs = world.observe(0);
    let cells = (g.width * g.height) as usize;
    let cell = |v: f32| {
        let i = v as i32;
        format!("({}, {})", i % g.width, i / g.width)
    };
    let mut at = 0;
    let mut parts = Vec::new();
    for f in &g.features {
        let take = match f {
            FeatureDef::SelfPosition { encoding: Encoding::Normalised } => 2,
            FeatureDef::SelfPosition { encoding: Encoding::OneHot } => cells,
            _ => 1,
        };
        let values = &obs.features[at..at + take];
        at += take;
        parts.push(match f {
            FeatureDef::SelfPosition { encoding: Encoding::Index } => format!("raft {}", cell(values[0])),
            FeatureDef::SelfPosition { .. } => {
                let (x, y) = world.position(0);
                format!("raft ({x}, {y})")
            }
            FeatureDef::AgentPosition { agent } => format!("{} {}", display_name(agent), cell(values[0])),
            FeatureDef::AgentDirection { agent } => {
                let way = ["here", "to the north", "to the east", "to the south", "to the west"];
                format!("{} {}", display_name(agent), way[values[0] as usize % 5])
            }
            FeatureDef::PreviousAction => {
                let moves = ["none yet", "north", "east", "south", "west"];
                format!("last move {}", moves[values[0] as usize % 5])
            }
        });
    }
    parts.join(" · ")
}

/// Level errors as one line for the player.
pub fn errors_text(errors: &[rl_envs::LevelError]) -> String {
    errors.iter().map(ToString::to_string).collect::<Vec<_>>().join(" · ")
}

/// The share of test voyages an experiment measure counts, if the map defines it.
pub fn measured_rate(measure: rl_tutorial::Measure, r: &ExperimentResult) -> Option<f32> {
    let n = r.tested.max(1) as f32;
    match measure {
        rl_tutorial::Measure::ReachGoal => Some(r.reached as f32 / n),
        rl_tutorial::Measure::ShortestRoute => r.shortest.map(|s| s as f32 / n),
    }
}

/// Stars for training efficiency: one per threshold the episode count is within.
pub fn stars(episodes_trained: u32, thresholds: &[u32]) -> u32 {
    thresholds.iter().filter(|&&t| episodes_trained <= t).count() as u32
}

/// The trained agent, and the walkthrough episode if one is open.
#[derive(Clone)]
struct Trainer {
    core: TabularTrainer,
    lesson: Option<Lesson>,
}

impl std::ops::Deref for Trainer {
    type Target = TabularTrainer;
    fn deref(&self) -> &TabularTrainer {
        &self.core
    }
}

impl std::ops::DerefMut for Trainer {
    fn deref_mut(&mut self) -> &mut TabularTrainer {
        &mut self.core
    }
}

/// A recorded voyage being played back on the arena, in its own world.
#[derive(Clone)]
struct Replay {
    world: GridWorld,
    actions: Vec<u32>,
    next: usize,
}

/// A training episode played move by move for the walkthrough.
#[derive(Clone)]
struct Lesson {
    world: GridWorld,
    episode: TrainingEpisode,
    number: u32,
    last: Option<LessonMove>,
    /// The learner and its random stream before this episode began: what is
    /// saved while the episode is unfinished, so saves stay on episode boundaries.
    checkpoint: (QLearner, Rng),
}

impl Lesson {
    fn in_progress(&self) -> bool {
        self.episode.finished().is_none()
    }
}

impl Trainer {
    /// One walkthrough move. Records the episode on the learning curve when it ends.
    fn lesson_move(&mut self) {
        let lesson = self.lesson.as_mut().expect("a lesson episode is under way");
        let from = lesson.world.position(0);
        let trace = lesson.episode.step(&mut lesson.world, &mut self.core.learner, &mut self.core.rng);
        let ActionSpace::Discrete { labels, .. } = &lesson.world.spec().agents[0].action;
        lesson.last = Some(LessonMove {
            action_label: labels.get(trace.action as usize).cloned().unwrap_or_default(),
            from,
            to: lesson.world.position(0),
            trace,
        });
        if let Some(done) = lesson.episode.finished() {
            self.core.curve.push(done.to_batch());
        }
    }

    /// Finishes a walkthrough episode left part-way, so bulk training continues
    /// from an episode boundary exactly as if it had trained all along.
    fn finish_lesson(&mut self) {
        while self.lesson.as_ref().is_some_and(Lesson::in_progress) {
            self.lesson_move();
        }
        self.lesson = None;
    }
}

#[derive(Default)]
pub struct SimCore {
    level_id: String,
    level_hash: u64,
    level_source: &'static str,
    world: Option<GridWorld>,
    replay: Option<Replay>,
    objective: Option<Objective>,
    random_baseline: Option<Baseline>,
    trainer: Option<Trainer>,
    policy_rng: Option<Rng>,
    sailed_by: Option<SailedBy>,
    seed: u64,
    episode: u32,
    last_action: Option<String>,
    last_reward: Option<f32>,
    episode_return: f32,
    outcome: Option<Outcome>,
    log: Vec<String>,
}

fn no_level() -> SimResponse {
    SimResponse::Error("no level loaded".into())
}

impl SimCore {
    pub fn handle(&mut self, request: SimRequest) -> SimResponse {
        match request {
            SimRequest::Batch { first_seed, episodes } => return self.batch(first_seed, episodes),
            SimRequest::Deploy => return self.deploy(),
            SimRequest::Load { level_id, seed } => {
                if let Err(e) = self.load(&level_id) {
                    return SimResponse::Error(e);
                }
                self.reset(seed);
            }
            SimRequest::Reset { seed } => {
                if self.world.is_none() {
                    return no_level();
                }
                self.reset(seed);
            }
            SimRequest::Step { action } => {
                let Some(world) = &self.world else { return no_level() };
                if world.grid().pilot == Pilot::Agent {
                    self.log = vec!["steering is off · the raft is the agent's job now".into()];
                } else {
                    self.step(action, SailedBy::Player);
                }
            }
            SimRequest::AgentStep { .. } if self.replay.is_some() => self.replay_step(),
            SimRequest::AgentStep { trained } => {
                let Some(world) = &self.world else { return no_level() };
                if world.grid().pilot == Pilot::Player {
                    self.log = vec!["in this level you are the pilot".into()];
                } else {
                    let observation = world.observe(0);
                    let mask = world.action_mask(0);
                    let rng = self.policy_rng.as_mut().expect("set on reset");
                    let (action, by) = match (&self.trainer, trained) {
                        (Some(t), true) => {
                            (GreedyPolicy(&t.learner.table).act(&observation, &mask, rng), SailedBy::TrainedAgent)
                        }
                        _ => (RandomAgent.act(&observation, &mask, rng), SailedBy::RandomAgent),
                    };
                    self.step(action.discrete(), by);
                }
            }
            SimRequest::Configure { config, rewards, observation, wind, training_seed } => {
                if self.world.is_none() {
                    return no_level();
                }
                if let Err(e) = self.configure(config, PlayerEdits { rewards, observation, wind }, training_seed) {
                    return SimResponse::Error(e);
                }
                let seed = self.seed;
                self.reset(seed);
            }
            SimRequest::Train { episodes } => {
                if self.world.is_none() {
                    return no_level();
                }
                if let Err(e) = self.train(episodes) {
                    return SimResponse::Error(e);
                }
                return SimResponse::Trained(Box::new(self.snapshot()));
            }
            SimRequest::Lesson { mode } => {
                if self.world.is_none() {
                    return no_level();
                }
                if let Err(e) = self.lesson(mode) {
                    return SimResponse::Error(e);
                }
                return SimResponse::Lesson(Box::new(self.snapshot()));
            }
            SimRequest::Experiment { config, adopt } => return self.experiment(&config, adopt),
            SimRequest::Outcomes { cell, trials } => return self.outcomes(cell, trials),
            SimRequest::Replay { seed, actions, edits } => {
                let Some(world) = &self.world else { return no_level() };
                let mut world = world.clone();
                if let Err(e) = world.apply_edits(&edits, EditMode::Strict) {
                    return SimResponse::Error(errors_text(&e));
                }
                self.reset(seed);
                world.reset(seed);
                self.replay = Some(Replay { world, actions, next: 0 });
                self.log = vec![format!("a recorded voyage · seed {seed} · played back from its action log")];
            }
            SimRequest::Export => {
                return match self.export() {
                    Some(save) => SimResponse::Export(Box::new(save)),
                    None => SimResponse::Error("there is no raft to save".into()),
                };
            }
            SimRequest::Restore(save) => {
                if self.world.is_none() {
                    return no_level();
                }
                if let Err(e) = self.restore(*save) {
                    return SimResponse::Error(e);
                }
            }
        }
        SimResponse::Snapshot(Box::new(self.snapshot()))
    }

    /// Runs the random agent over consecutive seeds without touching the
    /// live episode.
    pub fn batch(&self, first_seed: u64, episodes: u32) -> SimResponse {
        let Some(world) = &self.world else { return no_level() };
        if world.grid().pilot != Pilot::Agent {
            return SimResponse::Error("batch runs are for agent-piloted levels".into());
        }
        let episodes = episodes.clamp(1, MAX_BATCH_EPISODES);
        let mut probe = world.clone();
        let results =
            (0..u64::from(episodes)).map(|i| random_episode(&mut probe, first_seed.wrapping_add(i))).collect();
        let best = world.grid().best_return(0).map(|(r, _)| r);
        match BatchReport::new("random agent", results, best) {
            Some(report) => SimResponse::Batch(Box::new(report)),
            None => SimResponse::Error("empty batch".into()),
        }
    }

    fn new_trainer(world: &GridWorld, config: QLearningConfig, training_seed: u64) -> Option<Trainer> {
        if world.grid().pilot != Pilot::Agent {
            return None;
        }
        let states = tabular_states(world)?;
        let config = QLearningBounds::of(world).clamp(config);
        Some(Trainer { core: TabularTrainer::new(states, 4, config, training_seed), lesson: None })
    }

    /// Applies edits within the level's bounds and starts a fresh agent.
    fn configure(&mut self, config: QLearningConfig, edits: PlayerEdits, training_seed: u64) -> Result<(), String> {
        let world = self.world.as_mut().expect("checked");
        world.apply_edits(&edits, EditMode::Clamp).map_err(|e| errors_text(&e))?;
        self.trainer = Self::new_trainer(world, config, training_seed);
        if self.trainer.is_none() {
            return Err("this level has no trainable agent".into());
        }
        Ok(())
    }

    fn train(&mut self, episodes: u32) -> Result<(), String> {
        let world = self.world.as_mut().expect("checked");
        let budget = world.grid().budget.as_ref().map(|b| b.episodes);
        let trainer = self.trainer.as_mut().ok_or("this level has no trainable agent")?;
        // A walkthrough episode under way is finished first and counts as one of the episodes.
        let mut episodes = episodes;
        if trainer.lesson.as_ref().is_some_and(Lesson::in_progress) {
            trainer.finish_lesson();
            episodes = episodes.saturating_sub(1);
        }
        trainer.lesson = None;
        let done = trainer.curve.len() as u32;
        let allowed = budget.map_or(episodes, |b| episodes.min(b.saturating_sub(done))).min(MAX_TRAIN_CHUNK);
        if allowed == 0 && episodes > 0 {
            return Err("the training budget is used up · start a new raft to train again".into());
        }
        // Train on a copy, so the live episode on screen is untouched.
        trainer.core.train(&mut world.clone(), allowed);
        Ok(())
    }

    /// The walkthrough: real training, one move or one episode at a time.
    fn lesson(&mut self, mode: LessonMode) -> Result<(), String> {
        let world = self.world.as_ref().expect("checked");
        let budget = world.grid().budget.as_ref().map(|b| b.episodes);
        let trainer = self.trainer.as_mut().ok_or("this level has no trainable agent")?;
        let under_way = trainer.lesson.as_ref().is_some_and(Lesson::in_progress);
        if !under_way {
            let done = trainer.curve.len() as u32;
            if budget.is_some_and(|b| done >= b) {
                return Err("the training budget is used up · start a new raft to train again".into());
            }
            let mut lesson_world = world.clone();
            let seed = trainer.seed_for(done);
            let episode = TrainingEpisode::start(&mut lesson_world, seed);
            trainer.lesson = Some(Lesson {
                world: lesson_world,
                episode,
                number: done + 1,
                last: None,
                checkpoint: (trainer.learner.clone(), trainer.rng.clone()),
            });
            if mode != LessonMode::Episode {
                return Ok(());
            }
        }
        match mode {
            LessonMode::Start => {}
            LessonMode::Move => trainer.lesson_move(),
            LessonMode::Episode => {
                while trainer.lesson.as_ref().is_some_and(Lesson::in_progress) {
                    trainer.lesson_move();
                }
            }
        }
        Ok(())
    }

    /// Runs an experiment. With `adopt`, its raft and its edits become the
    /// current ones, as if the player had trained it.
    fn experiment(&mut self, config: &ExperimentConfig, adopt: bool) -> SimResponse {
        if self.world.is_none() {
            return no_level();
        }
        if config.level.id != self.level_id {
            return SimResponse::Error(format!("this experiment is for another level: \"{}\"", config.level.id));
        }
        let mut prepared = match config.start(self.level_source) {
            Ok(p) => p,
            Err(e) => return SimResponse::Error(errors_text(&e)),
        };
        let run = prepared.run();
        if adopt {
            let world = self.world.as_mut().expect("checked");
            if let Err(e) = world.apply_edits(&config.edits, EditMode::Strict) {
                return SimResponse::Error(errors_text(&e));
            }
            self.trainer = Some(Trainer { core: prepared.trainer, lesson: None });
        }
        let seed = self.seed;
        self.reset(seed);
        SimResponse::Experiment {
            result: ExperimentResult::of(&run),
            run: Box::new(run),
            snapshot: Box::new(self.snapshot()),
        }
    }

    /// Tries each move `trials` times from `cell`, each try on its own seed,
    /// in the level's real wind: the sea every test sails in.
    fn outcomes(&self, cell: (i32, i32), trials: u32) -> SimResponse {
        let Some(world) = &self.world else { return no_level() };
        let g = world.grid();
        if !g.in_bounds(cell) || g.is_wall(cell) {
            return SimResponse::Error("choose an open cell to try moves from".into());
        }
        let trials = trials.clamp(1, 1_000);
        let ActionSpace::Discrete { labels, .. } = &world.spec().agents[0].action;
        let mut probe = rl_training::test_world(world);
        let slip = probe.grid().slip;
        let moves = (0..4u32)
            .map(|a| {
                let mut landings: Vec<Landing> = Vec::new();
                let mut total = 0.0;
                for k in 0..u64::from(trials) {
                    probe.reset(k + 1);
                    probe.place(0, cell);
                    let r = probe.step(&vec![rl_core::Action::Discrete(a)]);
                    total += r.agents[0].reward;
                    let to = probe.position(0);
                    let note = r
                        .info
                        .events
                        .iter()
                        .find_map(|e| match e {
                            StepEvent::Died { .. } => Some("whirlpool".to_string()),
                            StepEvent::Caught { .. } => Some("caught".to_string()),
                            StepEvent::GoalReached { .. } => Some(g.goal_name.clone()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    match landings.iter_mut().find(|l| l.cell == to) {
                        Some(l) => l.count += 1,
                        None => landings.push(Landing { cell: to, count: 1, note }),
                    }
                }
                landings.sort_by_key(|l| std::cmp::Reverse(l.count));
                MoveOutcomes {
                    action: labels.get(a as usize).cloned().unwrap_or_default(),
                    landings,
                    mean_reward: total / trials as f32,
                }
            })
            .collect();
        SimResponse::Outcomes(Box::new(OutcomeReport { cell, trials, slip, moves }))
    }

    /// The current raft, as of its last finished episode.
    fn export(&self) -> Option<SavedRaft> {
        let t = self.trainer.as_ref()?;
        let (learner, rng) = match &t.lesson {
            Some(l) if l.in_progress() => (l.checkpoint.0.clone(), l.checkpoint.1.clone()),
            _ => (t.learner.clone(), t.rng.clone()),
        };
        let world = self.world.as_ref()?;
        Some(SavedRaft {
            schema: SAVE_SCHEMA,
            level_id: self.level_id.clone(),
            level_hash: self.level_hash,
            rewards: self.reward_settings().into_iter().map(|r| (r.term, r.weight)).collect(),
            training_seed: t.training_seed,
            learner,
            rng,
            curve: t.curve.clone(),
            observation: Some(world.grid().features.clone()),
            spec_hash: Some(world.spec().agents[0].spec_hash()),
            wind: Some(world.grid().slip),
        })
    }

    /// Loads a saved raft. Refused if it belongs to another level or another
    /// version of this level, or if training has already started.
    fn restore(&mut self, save: SavedRaft) -> Result<(), String> {
        if save.schema != SAVE_SCHEMA || save.level_id != self.level_id || save.level_hash != self.level_hash {
            return Err("the saved raft was trained on a different version of this level".into());
        }
        if self.trainer.as_ref().is_some_and(|t| !t.curve.is_empty() || t.lesson.is_some()) {
            return Err("a raft is already training · the saved raft was not loaded".into());
        }
        let edits =
            PlayerEdits { rewards: save.rewards.clone(), observation: save.observation.clone(), wind: save.wind };
        self.configure(save.learner.config, edits, save.training_seed)?;
        let world = self.world.as_ref().expect("checked");
        if let Some(saved) = save.spec_hash
            && saved != world.spec().agents[0].spec_hash()
        {
            self.trainer = Self::new_trainer(world, QLearningConfig::default(), 1);
            return Err("the saved raft was trained to observe a different world · it was not loaded".into());
        }
        let trainer = self.trainer.as_mut().expect("configured");
        let fits = trainer.learner.config == save.learner.config
            && trainer.learner.table.states == save.learner.table.states
            && trainer.learner.table.actions == save.learner.table.actions;
        if !fits {
            self.trainer = Self::new_trainer(self.world.as_ref().expect("checked"), QLearningConfig::default(), 1);
            return Err("the saved raft does not fit this level's settings".into());
        }
        trainer.core.learner = save.learner;
        trainer.core.rng = save.rng;
        trainer.core.curve = save.curve;
        self.log = vec![format!("your saved raft is back · {} training episodes", trainer.curve.len())];
        Ok(())
    }

    fn deploy(&self) -> SimResponse {
        let Some(world) = &self.world else { return no_level() };
        let Some(trainer) = &self.trainer else {
            return SimResponse::Error("there is no trained raft to deploy".into());
        };
        let Some(GoalDef::Tasks(tasks)) = self.objective.as_ref().map(|o| &o.goal) else {
            return SimResponse::Error("this level has no training tasks".into());
        };
        let best_return = rl_training::shortest_route_return(world);
        // Deployment sails the level's own wind, whatever the raft trained in.
        let mut probe = rl_training::test_world(world);
        let mut voyages = None;
        let mut results = Vec::new();
        for (i, task) in tasks.iter().enumerate() {
            let TaskDef::TrainAgent { threshold, eval_episodes, eval_seeds: EvalSeeds::Fixed(first), success, .. } =
                task
            else {
                continue;
            };
            let played: Vec<PlayedEpisode> = (0..u64::from(*eval_episodes))
                .map(|k| {
                    let seed = first.wrapping_add(k);
                    play_episode(&mut probe, &mut GreedyPolicy(&trainer.learner.table), seed, &mut agent_rng(seed))
                })
                .collect();
            let successes = played.iter().filter(|e| succeeded(*success, e, best_return)).count() as u32;
            let needed = (threshold * *eval_episodes as f32).ceil() as u32;
            results.push(TaskResult {
                task: i,
                successes,
                needed,
                tested: *eval_episodes,
                passed: successes >= needed,
            });
            voyages.get_or_insert(played);
        }
        let episodes_trained = trainer.curve.len() as u32;
        SimResponse::Deploy(Box::new(DeployReport {
            episodes_trained,
            stars: world.grid().budget.as_ref().map(|b| stars(episodes_trained, &b.stars)),
            tasks: results,
            voyages: voyages.unwrap_or_default(),
            rewards: self.reward_settings(),
        }))
    }

    fn reward_settings(&self) -> Vec<RewardSetting> {
        let Some(world) = &self.world else { return Vec::new() };
        let g = world.grid();
        let editable = g.player_editable.reward.as_ref();
        g.reward_terms
            .iter()
            .map(|&t| RewardSetting {
                term: t.name().into(),
                label: g.reward_labels[t.index()].clone(),
                weight: g.weight(t),
                editable: editable.is_some_and(|e| e.terms.iter().any(|n| n == t.name())),
            })
            .collect()
    }

    fn load(&mut self, level_id: &str) -> Result<(), String> {
        let entry = content::level(level_id).ok_or_else(|| format!("unknown level \"{level_id}\""))?;
        let world = rl_envs::load_world(entry.level_source).map_err(|e| format!("{level_id}: {e}"))?;
        self.level_hash = level_hash(entry.level_source);
        self.level_source = entry.level_source;
        let grid = world.grid();
        let best_route = grid.best_return(0).map(|(episode_return, moves)| BestRoute { episode_return, moves });
        self.objective =
            grid.challenge.as_ref().map(|c| Objective { text: c.objective.clone(), goal: c.goal.clone(), best_route });
        self.random_baseline = (grid.pilot == Pilot::Agent).then(|| {
            let mut probe = world.clone();
            let eps: Vec<BatchEpisode> =
                (0..u64::from(BASELINE_EPISODES)).map(|s| random_episode(&mut probe, s)).collect();
            let n = eps.len() as f32;
            Baseline {
                episodes: BASELINE_EPISODES,
                success_rate: eps.iter().filter(|e| e.reached_goal).count() as f32 / n,
                mean_return: eps.iter().map(|e| e.episode_return).sum::<f32>() / n,
                mean_length: eps.iter().map(|e| e.length as f32).sum::<f32>() / n,
            }
        });
        self.trainer = Self::new_trainer(&world, QLearningConfig::default(), 1);
        self.world = Some(world);
        self.level_id = level_id.to_string();
        self.episode = 0;
        Ok(())
    }

    fn reset(&mut self, seed: u64) {
        self.replay = None;
        let world = self.world.as_mut().expect("checked");
        world.reset(seed);
        self.policy_rng = Some(agent_rng(seed));
        self.sailed_by = Some(SailedBy::Nobody);
        self.seed = seed;
        self.episode += 1;
        self.last_action = None;
        self.last_reward = None;
        self.episode_return = 0.0;
        self.outcome = Some(Outcome::Sailing);
        self.log = vec![format!("episode {} begins · seed {seed}", self.episode)];
    }

    /// The next action of a recorded voyage.
    fn replay_step(&mut self) {
        let replay = self.replay.as_mut().expect("checked");
        match replay.actions.get(replay.next).copied() {
            Some(action) => {
                replay.next += 1;
                self.step(action, SailedBy::Replay);
            }
            None => self.log = vec!["the recorded voyage is over".into()],
        }
    }

    /// The world on the arena: a recorded voyage's while one plays, else the live one.
    fn shown_world(&self) -> &GridWorld {
        self.replay.as_ref().map_or_else(|| self.world.as_ref().expect("checked"), |r| &r.world)
    }

    fn step(&mut self, action: u32, by: SailedBy) {
        let world = match &mut self.replay {
            Some(r) => &mut r.world,
            None => self.world.as_mut().expect("checked"),
        };
        if world.is_done() {
            self.log = vec!["the episode is over · press R to sail again".into()];
            return;
        }
        let ActionSpace::Discrete { labels, .. } = world.spec().agents[0].action.clone();
        let label = labels.get(action as usize).cloned();
        let result = world.step(&vec![rl_core::Action::Discrete(action)]);
        let r = &result.agents[0];
        let died = result.info.events.iter().any(|e| matches!(e, StepEvent::Died { .. }));
        let caught = result.info.events.iter().any(|e| matches!(e, StepEvent::Caught { agent: 0, .. }));
        let names: Vec<String> = world.grid().agent_names.iter().map(|n| display_name(n)).collect();
        let goal = world.grid().goal_name.clone();
        self.sailed_by = Some(by);
        self.last_action = label.clone();
        self.last_reward = Some(r.reward);
        self.episode_return += r.reward;
        self.outcome = Some(if died {
            Outcome::Died
        } else if caught {
            Outcome::Caught
        } else if r.terminated {
            Outcome::ReachedShore
        } else if r.truncated {
            Outcome::OutOfTime
        } else {
            Outcome::Sailing
        });
        self.log = Vec::new();
        match by {
            SailedBy::RandomAgent => self.log.push(format!("random agent chose {}", label.unwrap_or_default())),
            SailedBy::TrainedAgent => self.log.push(format!("trained raft chose {}", label.unwrap_or_default())),
            SailedBy::Replay => self.log.push(format!("recorded move: {}", label.unwrap_or_default())),
            _ => {}
        }
        self.log.extend(result.info.events.iter().map(|e| match e {
            StepEvent::Moved { agent: 0, to, .. } => format!("moved to ({}, {})", to.0, to.1),
            StepEvent::Moved { agent, to, .. } => format!("{} moves to ({}, {})", names[*agent], to.0, to.1),
            StepEvent::Caught { by, .. } => format!("caught by {}", names[*by]),
            StepEvent::Gust { actual, .. } => {
                format!("a gust turns the raft {}", labels.get(*actual as usize).map_or("aside", String::as_str))
            }
            StepEvent::BumpedWall { .. } => "bumped into the edge · the move is wasted".into(),
            StepEvent::GoalReached { .. } => format!("reached {goal}"),
            StepEvent::Died { .. } => "swallowed by a whirlpool".into(),
            StepEvent::OnHazard { .. } => "scraped the rocks".into(),
            StepEvent::OnLure { .. } => "drifting among the lotus".into(),
            StepEvent::Reward { term, value, .. } => format!("reward {value:+} ({term})"),
        }));
        match self.outcome {
            Some(Outcome::ReachedShore | Outcome::Died | Outcome::Caught) => {
                self.log.push(format!("episode over · return {}", self.episode_return))
            }
            Some(Outcome::OutOfTime) => self.log.push("out of time · the episode is truncated".into()),
            _ => {}
        }
    }

    fn training_status(&self) -> Option<TrainingStatus> {
        let world = self.world.as_ref()?;
        let t = self.trainer.as_ref()?;
        let g = world.grid();
        // The arrows show what the raft would do in each cell, with everything
        // else it observes as it is now: in the walkthrough's world while one is open.
        let seen = t.lesson.as_ref().map_or_else(|| self.shown_world(), |l| &l.world);
        let cells = (g.width * g.height) as u32;
        let policy = (0..cells)
            .filter(|&i| !g.walls[i as usize])
            .filter_map(|i| {
                let cell = ((i as i32) % g.width, (i as i32) / g.width);
                let state = seen.observe_at(0, cell).state_index.filter(|&s| s < t.learner.table.states)?;
                let row = t.learner.table.row(state);
                Some(CellPolicy { cell, values: [row[0], row[1], row[2], row[3]] })
            })
            .collect();
        Some(TrainingStatus {
            config: t.learner.config,
            bounds: QLearningBounds::of(world),
            rewards: self.reward_settings(),
            reward_range: g.player_editable.reward.as_ref().map(|r| r.range),
            training_seed: t.training_seed,
            episodes_trained: t.curve.len() as u32,
            budget: g.budget.as_ref().map(|b| b.episodes),
            table_size: t.learner.table.states * t.learner.table.actions,
            curve: t.curve.clone(),
            policy,
            wind: g.player_editable.wind.map(|range| WindStatus { slip: g.slip, range, level: g.level_slip }),
            observation: g.player_editable.observation.as_ref().map(|edit| ObservationStatus {
                features: g.features.clone(),
                allowed: edit
                    .allowed
                    .iter()
                    .map(|f| FeatureOption {
                        feature: f.clone(),
                        name: f.name(),
                        cardinality: g.feature_cardinality(f).unwrap_or(1),
                    })
                    .collect(),
            }),
        })
    }

    fn snapshot(&self) -> Snapshot {
        let world = self.shown_world();
        let spec = &world.spec().agents[0];
        let g = world.grid();
        Snapshot {
            level_id: self.level_id.clone(),
            title: g.title.clone(),
            goal_name: g.goal_name.clone(),
            render: world.render_model(),
            pilot: g.pilot,
            objective: self.objective.clone(),
            budget_episodes: g.budget.as_ref().map(|b| b.episodes),
            random_baseline: self.random_baseline,
            sailed_by: self.sailed_by.unwrap_or(SailedBy::Nobody),
            training: self.training_status(),
            seed: self.seed,
            episode: self.episode,
            step: world.step_count(),
            max_steps: g.max_steps,
            position: world.position(0),
            state_index: world.observe(0).state_index,
            state_text: describe_state(world),
            state_count: spec.observation.state_count(),
            last_action: self.last_action.clone(),
            last_reward: self.last_reward,
            episode_return: self.episode_return,
            outcome: self.outcome.unwrap_or(Outcome::Sailing),
            log: self.log.clone(),
            lesson: self.lesson_state(),
            replaying: self.replay.is_some(),
        }
    }

    fn lesson_state(&self) -> Option<LessonState> {
        let lesson = self.trainer.as_ref()?.lesson.as_ref()?;
        let w = &lesson.world;
        Some(LessonState {
            episode: lesson.number,
            seed: lesson.episode.seed(),
            render: w.render_model(),
            moves: w.step_count(),
            max_steps: w.grid().max_steps,
            episode_return: lesson.episode.episode_return(),
            state: lesson.episode.state(),
            cell: w.position(0),
            last: lesson.last.clone(),
            finished: lesson.episode.finished(),
            goal_name: w.grid().goal_name.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(r: SimResponse) -> Snapshot {
        match r {
            SimResponse::Snapshot(s) => *s,
            other => panic!("expected a snapshot, got {other:?}"),
        }
    }

    fn load(sim: &mut SimCore, level: &str) -> Snapshot {
        snap(sim.handle(SimRequest::Load { level_id: level.into(), seed: 7 }))
    }

    #[test]
    fn sail_level_zero_to_the_shore() {
        let mut sim = SimCore::default();
        let s = load(&mut sim, "l00_calypsos_loom");
        assert_eq!((s.episode, s.step, s.position, s.state_count), (1, 0, (0, 0), Some(25)));
        assert_eq!(s.pilot, Pilot::Player);
        let objective = s.objective.as_ref().unwrap();
        assert_eq!(objective.goal, rl_envs::GoalDef::BestReturn);
        assert_eq!(objective.best_route, Some(BestRoute { episode_return: 2.0, moves: 8 }));
        assert_eq!(s.random_baseline, None);
        let mut last = s;
        for action in [1, 1, 1, 1, 2, 2, 2, 2] {
            last = snap(sim.handle(SimRequest::Step { action }));
        }
        assert_eq!(last.outcome, Outcome::ReachedShore);
        assert_eq!(last.episode_return, 2.0);
        assert_eq!(last.sailed_by, SailedBy::Player);
        assert_eq!(last.last_action.as_deref(), Some("south"));
        let after = snap(sim.handle(SimRequest::Step { action: 0 }));
        assert_eq!(after.step, 8);
        let reset = snap(sim.handle(SimRequest::Reset { seed: 7 }));
        assert_eq!((reset.episode, reset.step, reset.episode_return, reset.sailed_by), (2, 0, 0.0, SailedBy::Nobody));
    }

    #[test]
    fn level_one_is_agent_piloted() {
        let mut sim = SimCore::default();
        let s = load(&mut sim, "l01_the_raft");
        assert_eq!(s.pilot, Pilot::Agent);
        let baseline = s.random_baseline.expect("agent levels measure the random agent");
        assert_eq!(baseline.episodes, BASELINE_EPISODES);
        assert!(baseline.mean_return < 2.0, "random play must fall short of the best return: {baseline:?}");

        // The player cannot steer.
        let steered = snap(sim.handle(SimRequest::Step { action: 1 }));
        assert_eq!((steered.step, steered.position), (0, (0, 0)));
        assert!(steered.log[0].contains("steering is off"));

        // The random agent can, and its episodes are reproducible from the seed.
        let run = |sim: &mut SimCore| {
            sim.handle(SimRequest::Reset { seed: 3 });
            let mut s = snap(sim.handle(SimRequest::AgentStep { trained: false }));
            while s.outcome == Outcome::Sailing {
                s = snap(sim.handle(SimRequest::AgentStep { trained: false }));
            }
            (s.step, s.episode_return, s.sailed_by)
        };
        let a = run(&mut sim);
        assert_eq!(a, run(&mut sim));
        assert_eq!(a.2, SailedBy::RandomAgent);
    }

    #[test]
    fn batch_best_seed_replays_exactly_in_live_play() {
        let mut sim = SimCore::default();
        load(&mut sim, "l01_the_raft");
        let SimResponse::Batch(report) = sim.handle(SimRequest::Batch { first_seed: 1, episodes: 200 }) else {
            panic!("expected a batch report");
        };
        assert_eq!(report.episodes.len(), 200);
        assert_eq!(report.first_seed, 1);
        let best = *report.best_episode();
        assert!(best.episode_return >= report.returns.max);

        sim.handle(SimRequest::Reset { seed: best.seed });
        let mut s = snap(sim.handle(SimRequest::AgentStep { trained: false }));
        while s.outcome == Outcome::Sailing {
            s = snap(sim.handle(SimRequest::AgentStep { trained: false }));
        }
        assert_eq!((s.seed, s.step, s.episode_return), (best.seed, best.length, best.episode_return));
    }

    #[test]
    fn batch_runs_only_on_agent_levels() {
        let mut sim = SimCore::default();
        load(&mut sim, "l00_calypsos_loom");
        assert!(matches!(sim.handle(SimRequest::Batch { first_seed: 0, episodes: 10 }), SimResponse::Error(_)));
    }

    fn configure(sim: &mut SimCore, rewards: &[(&str, f32)]) -> Snapshot {
        snap(sim.handle(SimRequest::Configure {
            config: QLearningConfig::default(),
            rewards: rewards.iter().map(|(n, w)| (n.to_string(), *w)).collect(),
            observation: None,
            wind: None,
            training_seed: 1,
        }))
    }

    fn train(sim: &mut SimCore, episodes: u32) -> Snapshot {
        match sim.handle(SimRequest::Train { episodes }) {
            SimResponse::Trained(s) => *s,
            other => panic!("expected a training snapshot, got {other:?}"),
        }
    }

    fn deploy(sim: &mut SimCore) -> DeployReport {
        match sim.handle(SimRequest::Deploy) {
            SimResponse::Deploy(r) => *r,
            other => panic!("expected a deploy report, got {other:?}"),
        }
    }

    #[test]
    fn train_and_deploy_level_one() {
        let mut sim = SimCore::default();
        let s = load(&mut sim, "l01_the_raft");
        let t = s.training.expect("agent levels have a trainer");
        assert_eq!((t.episodes_trained, t.table_size, t.budget), (0, 100, Some(500)));
        let s = train(&mut sim, 300);
        let t = s.training.unwrap();
        assert_eq!(t.curve.len(), 300);
        assert_eq!(t.policy.len(), 25);
        let report = deploy(&mut sim);
        assert_eq!(report.tasks.len(), 1);
        assert!(report.tasks[0].passed, "{report:?}");
        assert_eq!(report.stars, Some(1), "300 episodes is within 500 but not 200");
        // The trained raft sails the shortest route live, too.
        sim.handle(SimRequest::Reset { seed: 5 });
        let mut s = snap(sim.handle(SimRequest::AgentStep { trained: true }));
        while s.outcome == Outcome::Sailing {
            s = snap(sim.handle(SimRequest::AgentStep { trained: true }));
        }
        assert_eq!((s.step, s.episode_return, s.sailed_by), (8, 2.0, SailedBy::TrainedAgent));
    }

    #[test]
    fn training_respects_the_budget() {
        let mut sim = SimCore::default();
        load(&mut sim, "l01_the_raft");
        train(&mut sim, 450);
        let s = train(&mut sim, 450);
        assert_eq!(s.training.unwrap().episodes_trained, 500);
        assert!(matches!(sim.handle(SimRequest::Train { episodes: 1 }), SimResponse::Error(_)));
        // A new raft resets the budget.
        let s = configure(&mut sim, &[]);
        assert_eq!(s.training.unwrap().episodes_trained, 0);
    }

    #[test]
    fn reward_edits_are_clamped_and_limited_to_editable_terms() {
        let mut sim = SimCore::default();
        load(&mut sim, "l02_the_lotus_eaters");
        let s = configure(&mut sim, &[("lure", 500.0), ("hazard", -30.0)]);
        let rewards = s.training.unwrap().rewards;
        let weight = |t: &str| rewards.iter().find(|r| r.term == t).unwrap().weight;
        assert_eq!(weight("lure"), 100.0, "clamped to the level's range");
        assert_eq!(weight("hazard"), -30.0);
        // Level 1 allows no reward edits.
        load(&mut sim, "l01_the_raft");
        let s = configure(&mut sim, &[("step", 5.0)]);
        let rewards = s.training.unwrap().rewards;
        assert_eq!(rewards.iter().find(|r| r.term == "step").unwrap().weight, -1.0);
    }

    #[test]
    fn level_two_lotus_rewards_breed_a_lotus_eater() {
        let mut sim = SimCore::default();
        load(&mut sim, "l02_the_lotus_eaters");
        configure(&mut sim, &[("lure", 5.0)]);
        train(&mut sim, 500);
        train(&mut sim, 300);
        let report = deploy(&mut sim);
        let passed: Vec<usize> = report.tasks.iter().filter(|t| t.passed).map(|t| t.task).collect();
        assert_eq!(passed, vec![1], "only the lotus-eater task: {report:?}");
    }

    #[test]
    fn level_three_is_won_by_a_raft_that_sees_polyphemus() {
        use rl_envs::level::{Encoding, FeatureDef};
        let mut sim = SimCore::default();
        let s = load(&mut sim, "l03_cyclops_cave");
        let o = s.training.unwrap().observation.expect("the Observation Builder is open here");
        assert_eq!((o.rows(), o.allowed.len()), (49, 4));
        train(&mut sim, 1000);
        let blind = deploy(&mut sim);
        assert!(!blind.tasks[0].passed, "position only: {blind:?}");

        let eyes = vec![
            FeatureDef::SelfPosition { encoding: Encoding::Index },
            FeatureDef::AgentDirection { agent: "polyphemus".into() },
        ];
        let s = snap(sim.handle(SimRequest::Configure {
            config: QLearningConfig::default(),
            rewards: vec![],
            observation: Some(eyes),
            wind: None,
            training_seed: 1,
        }));
        assert_eq!(s.training.unwrap().table_size, 245 * 4);
        assert!(s.state_text.contains("Polyphemus to the"), "{}", s.state_text);
        train(&mut sim, 500);
        let report = deploy(&mut sim);
        assert!(report.tasks[0].passed, "{report:?}");
        assert_eq!(report.stars, Some(3), "500 episodes earns every star");
    }

    #[test]
    fn level_four_tests_in_the_real_wind() {
        let mut sim = SimCore::default();
        let s = load(&mut sim, "l04_bag_of_winds");
        let w = s.training.unwrap().wind.expect("the training wind is editable here");
        assert_eq!((w.slip, w.level, w.range), (0.2, 0.2, (0.0, 0.4)));

        // Trained in calm water, deployed in the real wind: lost too often.
        let s = snap(sim.handle(SimRequest::Configure {
            config: QLearningConfig::default(),
            rewards: vec![],
            observation: None,
            wind: Some(0.0),
            training_seed: 1,
        }));
        assert_eq!(s.training.unwrap().wind.unwrap().slip, 0.0);
        train(&mut sim, 800);
        let calm = deploy(&mut sim);
        assert!(!calm.tasks[0].passed, "{calm:?}");
        assert!(calm.voyages.iter().any(|v| v.outcome == rl_training::EpisodeOutcome::Died));

        // The move tester sails the real wind, even while the raft trains in calm water.
        let SimResponse::Outcomes(r) = sim.handle(SimRequest::Outcomes { cell: (3, 3), trials: 100 }) else {
            panic!("expected outcomes");
        };
        let east = &r.moves[1];
        assert_eq!(east.landings.len(), 3, "in the wind: east, north or a whirlpool");
        assert!(east.landings.iter().any(|l| l.note == "whirlpool" && l.count > 0));
        assert_eq!(east.landings.iter().map(|l| l.count).sum::<u32>(), 100);
        assert!(matches!(sim.handle(SimRequest::Outcomes { cell: (0, 0), trials: 10 }), SimResponse::Error(_)));
    }

    fn lesson(sim: &mut SimCore, mode: LessonMode) -> Snapshot {
        match sim.handle(SimRequest::Lesson { mode }) {
            SimResponse::Lesson(s) => *s,
            other => panic!("expected a lesson snapshot, got {other:?}"),
        }
    }

    #[test]
    fn walkthrough_moves_are_real_training() {
        // Walk through one episode and part of the next, then bulk-train the
        // rest: the raft must be identical to one trained in bulk all along.
        let mut stepped = SimCore::default();
        load(&mut stepped, "l01_the_raft");
        let s = lesson(&mut stepped, LessonMode::Start);
        let l = s.lesson.expect("the walkthrough has begun");
        assert_eq!((l.episode, l.seed, l.moves, l.last.is_none()), (1, 1, 0, true));
        let s = lesson(&mut stepped, LessonMode::Move);
        let m = s.lesson.unwrap().last.expect("one move made");
        let t = &m.trace;
        assert_eq!((t.step, t.state, m.from), (1, 0, (0, 0)));
        assert_eq!(t.update.old_value, t.row_before[t.action as usize]);
        let row = &s.training.as_ref().unwrap().policy[0].values;
        assert_eq!(row[t.action as usize], t.update.new_value, "the update landed in the Q-table");
        let s = lesson(&mut stepped, LessonMode::Episode);
        let l = s.lesson.as_ref().unwrap();
        assert_eq!(Some(l.episode_return), l.finished.map(|f| f.episode_return), "the final return stays shown");
        assert_eq!(s.training.as_ref().unwrap().episodes_trained, 1);
        for _ in 0..3 {
            lesson(&mut stepped, LessonMode::Move);
        }
        let s = train(&mut stepped, 99);
        assert!(s.lesson.is_none(), "bulk training closes the walkthrough episode");

        let mut bulk = SimCore::default();
        load(&mut bulk, "l01_the_raft");
        let b = train(&mut bulk, 100);
        assert_eq!(s.training, b.training);
    }

    #[test]
    fn walkthrough_respects_the_budget() {
        let mut sim = SimCore::default();
        load(&mut sim, "l01_the_raft");
        train(&mut sim, 500);
        assert!(matches!(sim.handle(SimRequest::Lesson { mode: LessonMode::Move }), SimResponse::Error(_)));
    }

    fn export(sim: &mut SimCore) -> SavedRaft {
        match sim.handle(SimRequest::Export) {
            SimResponse::Export(s) => *s,
            other => panic!("expected a saved raft, got {other:?}"),
        }
    }

    #[test]
    fn a_saved_raft_continues_training_exactly() {
        let mut first = SimCore::default();
        load(&mut first, "l02_the_lotus_eaters");
        configure(&mut first, &[("hazard", -30.0)]);
        train(&mut first, 120);
        // An unfinished walkthrough episode is not part of the save.
        lesson(&mut first, LessonMode::Move);
        let save = export(&mut first);
        assert_eq!(save.curve.len(), 120);
        let json = serde_json::to_string(&save).unwrap();

        let mut second = SimCore::default();
        load(&mut second, "l02_the_lotus_eaters");
        let restored = snap(second.handle(SimRequest::Restore(Box::new(serde_json::from_str(&json).unwrap()))));
        let t = restored.training.unwrap();
        assert_eq!(t.episodes_trained, 120);
        assert_eq!(t.rewards.iter().find(|r| r.term == "hazard").unwrap().weight, -30.0);

        let mut straight = SimCore::default();
        load(&mut straight, "l02_the_lotus_eaters");
        configure(&mut straight, &[("hazard", -30.0)]);
        train(&mut straight, 120);
        assert_eq!(train(&mut second, 80).training, train(&mut straight, 80).training);
    }

    #[test]
    fn a_saved_raft_is_refused_by_another_level_or_a_busy_trainer() {
        let mut sim = SimCore::default();
        load(&mut sim, "l01_the_raft");
        train(&mut sim, 10);
        let mut save = export(&mut sim);
        assert!(matches!(sim.handle(SimRequest::Restore(Box::new(save.clone()))), SimResponse::Error(_)));
        load(&mut sim, "l01_the_raft");
        save.level_hash ^= 1;
        assert!(matches!(sim.handle(SimRequest::Restore(Box::new(save.clone()))), SimResponse::Error(_)));
        save.level_hash ^= 1;
        load(&mut sim, "l02_the_lotus_eaters");
        assert!(matches!(sim.handle(SimRequest::Restore(Box::new(save))), SimResponse::Error(_)));
    }

    #[test]
    fn every_guided_experiment_shows_its_lesson() {
        let mut checked = 0;
        for entry in content::LEVELS {
            let tutorial = rl_tutorial::TutorialDef::parse(entry.tutorial_source).unwrap();
            for (i, step) in tutorial.steps.iter().enumerate() {
                let Some(def) = step.experiment() else { continue };
                let mut sim = SimCore::default();
                load(&mut sim, entry.id);
                let config = experiment_spec(entry.id, def).unwrap();
                let SimResponse::Experiment { result, snapshot, .. } =
                    sim.handle(SimRequest::Experiment { config: Box::new(config), adopt: true })
                else {
                    panic!("{} step {i}: the experiment did not run", entry.id);
                };
                assert_eq!(result.episodes_trained, def.episodes, "{} step {i}", entry.id);
                assert_eq!(snapshot.training.unwrap().episodes_trained, def.episodes);
                for e in &def.expect {
                    let rate = measured_rate(e.measure, &result)
                        .unwrap_or_else(|| panic!("{} step {i}: {:?} is undefined on this map", entry.id, e.measure));
                    assert!(e.holds(rate), "{} step {i}: {e:?} failed with {rate} ({result:?})", entry.id);
                }
                checked += 1;
            }
        }
        assert!(checked >= 2, "the Level 1 tutorial has two experiments");
    }

    #[test]
    fn errors_are_reported_not_panicked() {
        let mut sim = SimCore::default();
        assert!(matches!(sim.handle(SimRequest::Step { action: 0 }), SimResponse::Error(_)));
        assert!(matches!(sim.handle(SimRequest::AgentStep { trained: false }), SimResponse::Error(_)));
        assert!(matches!(sim.handle(SimRequest::Load { level_id: "nope".into(), seed: 0 }), SimResponse::Error(_)));
    }
}
