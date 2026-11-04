//! Headless episode running, replay and evaluation. Rendering is never needed.

pub mod episode;
pub mod runner;
pub mod stats;

pub use episode::{EpisodeOutcome, PlayedEpisode, TrainingEpisode, TransitionTrace, play_episode, train_episode};
pub use runner::{EpisodeRecord, EvalSummary, evaluate, replay, run_episode};
pub use stats::{BatchEpisode, BatchReport, Summary, moving_mean, running_mean};
