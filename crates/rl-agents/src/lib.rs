//! Agents. Each agent is a [`rl_core::PolicyProvider`].

pub mod q_learning;
pub mod random;

pub use q_learning::{GreedyPolicy, QLearner, QLearningConfig, QTable, UpdateTrace};
pub use random::RandomAgent;
