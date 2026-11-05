//! Playing and training single episodes for agent 0, with outcome tracking.

use rl_agents::{QLearner, UpdateTrace};
use rl_core::{Observation, PolicyProvider, Rng, StepEvent, StepResults, World};
use serde::{Deserialize, Serialize};

use crate::stats::BatchEpisode;

/// How an episode ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpisodeOutcome {
    ReachedGoal,
    Died,
    /// Caught by an opponent.
    Caught,
    OutOfTime,
}

/// One played episode, with the facts needed to judge its behaviour.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlayedEpisode {
    pub seed: u64,
    pub episode_return: f32,
    pub length: u32,
    pub outcome: EpisodeOutcome,
    /// Steps that ended on a hazard cell.
    pub hazard_steps: u32,
    /// Steps that ended on a lure cell.
    pub lure_steps: u32,
    /// Moves chosen at random by ε-greedy exploration (training only).
    #[serde(default)]
    pub explored: u32,
}

impl PlayedEpisode {
    pub fn to_batch(self) -> BatchEpisode {
        BatchEpisode {
            seed: self.seed,
            episode_return: self.episode_return,
            length: self.length,
            reached_goal: self.outcome == EpisodeOutcome::ReachedGoal,
            explored: self.explored,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct Tally {
    episode_return: f32,
    hazard_steps: u32,
    lure_steps: u32,
    explored: u32,
    died: bool,
    caught: bool,
}

impl Tally {
    fn add(&mut self, r: &StepResults) {
        self.episode_return += r.agents[0].reward;
        for e in &r.info.events {
            match e {
                StepEvent::OnHazard { agent: 0, .. } => self.hazard_steps += 1,
                StepEvent::OnLure { agent: 0, .. } => self.lure_steps += 1,
                StepEvent::Died { agent: 0, .. } => self.died = true,
                StepEvent::Caught { agent: 0, .. } => self.caught = true,
                _ => {}
            }
        }
    }

    fn finish<W: World + ?Sized>(self, world: &W, seed: u64, last: &StepResults) -> PlayedEpisode {
        let outcome = if self.died {
            EpisodeOutcome::Died
        } else if self.caught {
            EpisodeOutcome::Caught
        } else if last.agents[0].terminated {
            EpisodeOutcome::ReachedGoal
        } else {
            EpisodeOutcome::OutOfTime
        };
        PlayedEpisode {
            seed,
            episode_return: self.episode_return,
            length: world.step_count(),
            outcome,
            hazard_steps: self.hazard_steps,
            lure_steps: self.lure_steps,
            explored: self.explored,
        }
    }
}

/// An agent's own random stream for an episode on `seed`. Live play, batch
/// runs, deployment and experiment evaluation all use it, so any seed
/// replays exactly.
pub fn agent_rng(seed: u64) -> Rng {
    Rng::new(seed).fork(0x5241_4654) // "RAFT"
}

/// Plays one episode on `seed` with `policy` and its own random stream,
/// recording every action, so the episode can be replayed from the seed and
/// the action log alone.
pub fn play_recorded<W: World + ?Sized>(world: &mut W, policy: &mut dyn PolicyProvider, seed: u64) -> RecordedEpisode {
    let mut rng = agent_rng(seed);
    let mut obs: Observation = world.reset(seed).swap_remove(0);
    let mut tally = Tally::default();
    let mut actions = Vec::new();
    loop {
        let action = policy.act(&obs, &world.action_mask(0), &mut rng);
        actions.push(action.discrete());
        let r = world.step(&vec![action]);
        tally.add(&r);
        if r.done() {
            return RecordedEpisode { played: tally.finish(world, seed, &r), actions };
        }
        obs = r.agents[0].observation.clone();
    }
}

/// A played episode with its action log: the seed and the actions rebuild it exactly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecordedEpisode {
    pub played: PlayedEpisode,
    pub actions: Vec<u32>,
}

/// Plays one episode of a single-agent world with `policy`.
pub fn play_episode<W: World + ?Sized>(
    world: &mut W,
    policy: &mut dyn PolicyProvider,
    seed: u64,
    rng: &mut Rng,
) -> PlayedEpisode {
    let mut obs: Observation = world.reset(seed).swap_remove(0);
    let mut tally = Tally::default();
    loop {
        let action = policy.act(&obs, &world.action_mask(0), rng);
        let r = world.step(&vec![action]);
        tally.add(&r);
        if r.done() {
            return tally.finish(world, seed, &r);
        }
        obs = r.agents[0].observation.clone();
    }
}

/// Everything one training move did: the walkthrough shows these values
/// line by line, so what is taught is exactly what trains.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransitionTrace {
    /// Move number within the episode, from 1.
    pub step: u32,
    pub state: u32,
    pub action: u32,
    /// Whether ε-greedy picked a random move rather than a best one.
    pub explored: bool,
    pub reward: f32,
    pub next_state: u32,
    /// Q-values of `state` before the update.
    pub row_before: Vec<f32>,
    /// Q-values of `next_state` used for the target, before the update.
    pub next_row: Vec<f32>,
    /// `max_a' Q(s', a')`, or `None` when the episode terminated and nothing is bootstrapped.
    pub next_max: Option<f32>,
    pub terminated: bool,
    pub truncated: bool,
    pub update: UpdateTrace,
}

/// A training episode played one move at a time. [`train_episode`] is this
/// stepper run to the end, so stepping and bulk training are identical.
#[derive(Clone, Debug)]
pub struct TrainingEpisode {
    seed: u64,
    state: u32,
    tally: Tally,
    finished: Option<PlayedEpisode>,
}

fn state_of(o: &Observation) -> u32 {
    o.state_index.expect("Q-learning needs a discrete observation")
}

impl TrainingEpisode {
    /// Resets `world` with `seed` and gets ready for the first move.
    pub fn start<W: World + ?Sized>(world: &mut W, seed: u64) -> Self {
        let state = state_of(&world.reset(seed)[0]);
        Self { seed, state, tally: Tally::default(), finished: None }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The current state s.
    pub fn state(&self) -> u32 {
        self.state
    }

    /// Return so far, or the final return once the episode is over.
    pub fn episode_return(&self) -> f32 {
        self.finished.map_or(self.tally.episode_return, |f| f.episode_return)
    }

    /// The finished episode, once it has terminated or run out of time.
    pub fn finished(&self) -> Option<PlayedEpisode> {
        self.finished
    }

    /// One move: ε-greedy choice, environment step, Q-learning update.
    /// `world` must be the world this episode was started on.
    pub fn step<W: World + ?Sized>(&mut self, world: &mut W, learner: &mut QLearner, rng: &mut Rng) -> TransitionTrace {
        assert!(self.finished.is_none(), "the episode is over");
        let state = self.state;
        let mask = world.action_mask(0);
        let (action, explored) = learner.choose(state, &mask, rng);
        self.tally.explored += u32::from(explored);
        let row_before = learner.table.row(state).to_vec();
        let r = world.step(&vec![rl_core::Action::Discrete(action)]);
        self.tally.add(&r);
        let result = &r.agents[0];
        let next_state = state_of(&result.observation);
        let next_mask = world.action_mask(0);
        let next_row = learner.table.row(next_state).to_vec();
        let bootstrap = (!result.terminated).then_some((next_state, &next_mask));
        let next_max = bootstrap.map(|(s2, m)| learner.table.max_value(s2, m));
        let update = learner.update(state, action, result.reward, bootstrap);
        let trace = TransitionTrace {
            step: world.step_count(),
            state,
            action,
            explored,
            reward: result.reward,
            next_state,
            row_before,
            next_row,
            next_max,
            terminated: result.terminated,
            truncated: result.truncated,
            update,
        };
        if r.done() {
            self.finished = Some(std::mem::take(&mut self.tally).finish(world, self.seed, &r));
        }
        self.state = next_state;
        trace
    }
}

/// Trains `learner` on one episode with ε-greedy exploration.
pub fn train_episode<W: World + ?Sized>(
    world: &mut W,
    learner: &mut QLearner,
    seed: u64,
    rng: &mut Rng,
) -> PlayedEpisode {
    let mut episode = TrainingEpisode::start(world, seed);
    loop {
        episode.step(world, learner, rng);
        if let Some(done) = episode.finished() {
            return done;
        }
    }
}
