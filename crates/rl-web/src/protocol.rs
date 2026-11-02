//! Messages between the UI thread and the simulation worker.

use rl_core::RenderModel;
pub use rl_envs::Pilot;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SimRequest {
    /// Loads a bundled level and starts episode 1 with `seed`.
    Load { level_id: String, seed: u64 },
    /// Starts a new episode with `seed`.
    Reset { seed: u64 },
    /// The player steers agent 0. Ignored in agent-piloted levels.
    Step { action: u32 },
    /// The level's agent takes one step. Until training exists (Milestone 1)
    /// this is the random agent.
    AgentStep,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SimResponse {
    Snapshot(Box<Snapshot>),
    Error(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    Sailing,
    ReachedShore,
    OutOfTime,
}

/// Who has been steering in the current episode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SailedBy {
    Nobody,
    Player,
    RandomAgent,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Objective {
    pub text: String,
    pub goal: ObjectiveGoal,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ObjectiveGoal {
    /// Reach the goal with this return, the best possible (`moves` moves).
    BestReturn { target: f32, moves: u32 },
    /// A trained agent must succeed in `threshold` of `eval_episodes` test episodes.
    SuccessRate { threshold: f32, eval_episodes: u32 },
}

/// How a baseline agent performs over many episodes.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Baseline {
    pub episodes: u32,
    pub success_rate: f32,
    pub mean_return: f32,
    pub mean_length: f32,
}

/// Everything the UI shows about the current moment of the RL loop.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    pub level_id: String,
    pub title: String,
    pub render: RenderModel,
    pub pilot: Pilot,
    pub objective: Option<Objective>,
    pub budget_episodes: Option<u32>,
    /// The random agent's record on this level, for agent-piloted levels.
    pub random_baseline: Option<Baseline>,
    pub sailed_by: SailedBy,
    pub seed: u64,
    pub episode: u32,
    pub step: u32,
    pub max_steps: u32,
    pub position: (i32, i32),
    pub state_index: Option<u32>,
    /// Number of distinct states (the Q-table row count), if discrete.
    pub state_count: Option<u64>,
    pub last_action: Option<String>,
    pub last_reward: Option<f32>,
    pub episode_return: f32,
    pub outcome: Outcome,
    /// Human-readable lines describing what happened in the last step.
    pub log: Vec<String>,
}
