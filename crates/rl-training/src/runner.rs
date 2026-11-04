//! Episode runner and replay.
//!
//! An episode is recorded as its seed plus the joint action at every step.
//! Because worlds are deterministic, that is enough to rebuild every
//! observation and reward, so replays and offline datasets cost almost nothing.

use rl_core::{JointAction, PolicyProvider, Rng, StepResults, World};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EpisodeRecord {
    pub seed: u64,
    pub actions: Vec<JointAction>,
    /// Undiscounted return per agent.
    pub returns: Vec<f32>,
    pub length: u32,
    /// True if every agent terminated (e.g. reached the goal) rather than ran out of time.
    pub terminated: bool,
}

/// Runs one episode with one policy per agent. `policy_seed` drives the
/// policies' own randomness, separately from the world's seed.
pub fn run_episode<W: World + ?Sized>(
    world: &mut W,
    policies: &mut [&mut dyn PolicyProvider],
    seed: u64,
    policy_seed: u64,
) -> EpisodeRecord {
    let n = world.spec().agents.len();
    assert_eq!(policies.len(), n, "one policy per agent is required");
    let mut rngs: Vec<Rng> = (0..n as u64).map(|i| Rng::new(policy_seed).fork(i)).collect();
    let mut observations = world.reset(seed);
    let mut record = EpisodeRecord { seed, actions: Vec::new(), returns: vec![0.0; n], length: 0, terminated: false };
    loop {
        let actions: JointAction =
            (0..n).map(|i| policies[i].act(&observations[i], &world.action_mask(i), &mut rngs[i])).collect();
        let result = world.step(&actions);
        record.actions.push(actions);
        record.length += 1;
        for (i, r) in result.agents.iter().enumerate() {
            record.returns[i] += r.reward;
        }
        if result.done() {
            record.terminated = result.agents.iter().all(|r| r.terminated);
            return record;
        }
        observations = result.agents.into_iter().map(|r| r.observation).collect();
    }
}

/// Rebuilds every step of a recorded episode by re-simulating it.
pub fn replay<W: World + ?Sized>(world: &mut W, record: &EpisodeRecord) -> Vec<StepResults> {
    world.reset(record.seed);
    record.actions.iter().map(|a| world.step(a)).collect()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvalSummary {
    pub episodes: u32,
    pub success_rate: f32,
    pub mean_return: f32,
    pub mean_length: f32,
}

/// Deterministic evaluation of agent 0 on seeds `first_seed..first_seed + episodes`.
pub fn evaluate<W: World + ?Sized>(
    world: &mut W,
    policies: &mut [&mut dyn PolicyProvider],
    first_seed: u64,
    episodes: u32,
) -> EvalSummary {
    let mut successes = 0;
    let mut total_return = 0.0;
    let mut total_length = 0;
    for e in 0..u64::from(episodes) {
        let seed = first_seed + e;
        let r = run_episode(world, policies, seed, seed);
        successes += u32::from(r.terminated);
        total_return += r.returns[0];
        total_length += r.length;
    }
    let n = episodes.max(1) as f32;
    EvalSummary {
        episodes,
        success_rate: successes as f32 / n,
        mean_return: total_return / n,
        mean_length: total_length as f32 / n,
    }
}
