//! Headless episode running, replay and evaluation. Rendering is never needed.

pub mod runner;

pub use runner::{EpisodeRecord, EvalSummary, evaluate, replay, run_episode};
