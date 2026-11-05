//! A tabular Q-learning agent in training. The browser's live raft and every
//! experiment train through this, so the same settings always give the same agent.

use rl_agents::{QLearner, QLearningConfig};
use rl_core::{Rng, World};

use crate::episode::train_episode;
use crate::stats::BatchEpisode;

/// The learner, its own random stream and its learning curve. Training
/// episode `k` (from 0) sails on seed `training_seed + k`.
#[derive(Clone, Debug, PartialEq)]
pub struct TabularTrainer {
    pub learner: QLearner,
    pub training_seed: u64,
    pub rng: Rng,
    /// Every training episode so far.
    pub curve: Vec<BatchEpisode>,
}

impl TabularTrainer {
    pub fn new(states: u32, actions: u32, config: QLearningConfig, training_seed: u64) -> Self {
        Self {
            learner: QLearner::new(states, actions, config),
            training_seed,
            rng: Rng::new(training_seed),
            curve: Vec::new(),
        }
    }

    /// The seed of training episode `index`, counted from 0.
    pub fn seed_for(&self, index: u32) -> u64 {
        self.training_seed.wrapping_add(u64::from(index))
    }

    pub fn episodes_trained(&self) -> u32 {
        self.curve.len() as u32
    }

    /// Trains `episodes` more episodes on `world`, which is reset for each one.
    pub fn train<W: World + ?Sized>(&mut self, world: &mut W, episodes: u32) {
        for _ in 0..episodes {
            let seed = self.seed_for(self.episodes_trained());
            let e = train_episode(world, &mut self.learner, seed, &mut self.rng);
            self.curve.push(e.to_batch());
        }
    }
}
