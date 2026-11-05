//! Headless episode running, training, replay, evaluation and experiments.
//! Rendering is never needed.

pub mod episode;
pub mod experiment;
pub mod runner;
pub mod stats;
pub mod trainer;

pub use episode::{
    EpisodeOutcome, PlayedEpisode, RecordedEpisode, TrainingEpisode, TransitionTrace, agent_rng, play_episode,
    play_recorded, train_episode,
};
pub use experiment::{
    Algorithm, EXPERIMENT_SCHEMA, Evaluation, EvaluationReport, ExperimentConfig, ExperimentRun, LevelRef,
    PreparedExperiment, QLearningBounds, run_experiment, shortest_route_return, succeeded, tabular_states,
};
pub use runner::{EpisodeRecord, EvalSummary, evaluate, replay, run_episode};
pub use stats::{BatchEpisode, BatchReport, Summary, moving_mean, running_mean};
pub use trainer::TabularTrainer;
