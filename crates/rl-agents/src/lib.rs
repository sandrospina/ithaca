//! Agents. Each agent is a [`rl_core::PolicyProvider`].

pub mod random;

pub use random::RandomAgent;
