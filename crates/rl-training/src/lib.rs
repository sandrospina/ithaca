//! Headless episode running, replay and evaluation. Rendering is never needed.

pub mod runner;
pub mod stats;

pub use runner::{EpisodeRecord, EvalSummary, evaluate, replay, run_episode};
pub use stats::{BatchEpisode, BatchReport, Summary};
