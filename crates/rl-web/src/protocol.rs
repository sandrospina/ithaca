//! Messages between the UI thread and the simulation worker.

pub use rl_agents::{QLearner, QLearningConfig};
use rl_core::RenderModel;
use rl_core::Rng;
pub use rl_envs::{GoalDef, Pilot};
pub use rl_training::{BatchEpisode, BatchReport, EpisodeOutcome, PlayedEpisode, TransitionTrace};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SimRequest {
    /// Loads a bundled level and starts episode 1 with `seed`.
    Load { level_id: String, seed: u64 },
    /// Starts a new episode with `seed`.
    Reset { seed: u64 },
    /// The player steers agent 0. Ignored in agent-piloted levels.
    Step { action: u32 },
    /// An agent takes one step: the random agent, or the trained raft.
    AgentStep { trained: bool },
    /// Starts a fresh trained agent with these settings. Values outside the
    /// level's editing bounds are clamped; terms the player may not edit are ignored.
    Configure { config: QLearningConfig, rewards: Vec<(String, f32)>, training_seed: u64 },
    /// Trains the current agent for up to `episodes` more episodes, within the budget.
    Train { episodes: u32 },
    /// Tests the trained agent's greedy policy on each training task's test seeds.
    Deploy,
    /// Runs the level's agent headlessly on seeds `first_seed..first_seed + episodes`.
    Batch { first_seed: u64, episodes: u32 },
    /// The live walkthrough: training one move or one episode at a time. These
    /// are real training episodes of the current raft and count toward the budget.
    Lesson { mode: LessonMode },
    /// A tutorial's guided experiment: a new raft with pinned settings, trained
    /// and then tested on the level's test seeds.
    Experiment(ExperimentSpec),
    /// Asks for the current raft in a form that can be saved.
    Export,
    /// Replaces the current raft with a saved one, if it was trained on this
    /// exact level file and nothing has been trained since loading.
    Restore(Box<SavedRaft>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LessonMode {
    /// Begins the next training episode if none is under way.
    Start,
    /// One move: choose, act, update. Begins an episode if none is under way.
    Move,
    /// Plays the current episode to its end, beginning one if needed.
    Episode,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperimentSpec {
    pub config: QLearningConfig,
    pub rewards: Vec<(String, f32)>,
    pub training_seed: u64,
    pub episodes: u32,
}

/// How a guided experiment's raft did on the test seeds.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperimentResult {
    pub episodes_trained: u32,
    pub tested: u32,
    pub reached: u32,
    /// Voyages on the shortest route, when the map defines one.
    pub shortest: Option<u32>,
    pub mean_return: f32,
    pub mean_length: f32,
}

/// Version of [`SavedRaft`]; older saves are ignored.
pub const SAVE_SCHEMA: u32 = 1;

/// A trained raft as stored in the browser: enough to continue training
/// exactly where it stopped.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedRaft {
    pub schema: u32,
    pub level_id: String,
    /// Hash of the level file it was trained on. A changed level invalidates the raft.
    pub level_hash: u64,
    pub rewards: Vec<(String, f32)>,
    pub training_seed: u64,
    pub learner: QLearner,
    pub rng: Rng,
    pub curve: Vec<BatchEpisode>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SimResponse {
    Snapshot(Box<Snapshot>),
    /// A snapshot after a training chunk, so the UI can request the next one.
    Trained(Box<Snapshot>),
    Batch(Box<BatchReport>),
    Deploy(Box<DeployReport>),
    /// A snapshot after a walkthrough request, so the UI can schedule the next one.
    Lesson(Box<Snapshot>),
    Experiment {
        result: ExperimentResult,
        snapshot: Box<Snapshot>,
    },
    Export(Box<SavedRaft>),
    Error(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    Sailing,
    ReachedShore,
    Died,
    OutOfTime,
}

/// Who has been steering in the current episode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SailedBy {
    Nobody,
    Player,
    RandomAgent,
    TrainedAgent,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Objective {
    pub text: String,
    pub goal: GoalDef,
    /// The best possible return and the moves it takes, when a goal is reachable.
    pub best_route: Option<BestRoute>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BestRoute {
    pub episode_return: f32,
    pub moves: u32,
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
    /// The trained agent, on agent-piloted levels.
    pub training: Option<TrainingStatus>,
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
    /// The walkthrough episode, once the walkthrough has been opened.
    pub lesson: Option<LessonState>,
}

/// The walkthrough's training episode: shown in its own copy of the world,
/// separate from the voyage on the arena.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LessonState {
    /// Training episode number, from 1.
    pub episode: u32,
    pub seed: u64,
    pub render: RenderModel,
    pub moves: u32,
    pub max_steps: u32,
    pub episode_return: f32,
    /// The current state s and its cell.
    pub state: u32,
    pub cell: (i32, i32),
    /// The move just made, if any in this episode.
    pub last: Option<LessonMove>,
    pub finished: Option<PlayedEpisode>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LessonMove {
    pub trace: TransitionTrace,
    pub action_label: String,
    pub from: (i32, i32),
    pub to: (i32, i32),
}

/// One reward term as the reward editor shows it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RewardSetting {
    pub term: String,
    pub label: String,
    pub weight: f32,
    pub editable: bool,
}

/// Inclusive bounds for each learning setting.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConfigBounds {
    pub alpha: (f32, f32),
    pub gamma: (f32, f32),
    pub epsilon: (f32, f32),
}

/// What the trained agent believes about one cell.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CellPolicy {
    pub cell: (i32, i32),
    /// Q-values for north, east, south, west.
    pub values: [f32; 4],
}

/// The current trained agent and its settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingStatus {
    pub config: QLearningConfig,
    pub bounds: ConfigBounds,
    pub rewards: Vec<RewardSetting>,
    pub reward_range: Option<(f32, f32)>,
    pub training_seed: u64,
    pub episodes_trained: u32,
    pub budget: Option<u32>,
    /// Number of Q-values in the table: states × actions.
    pub table_size: u32,
    /// Every training episode so far: the learning curve.
    pub curve: Vec<BatchEpisode>,
    /// Q-values per open cell, when states map one-to-one onto cells.
    pub policy: Vec<CellPolicy>,
}

/// The result of one training task on its test seeds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaskResult {
    pub task: usize,
    pub successes: u32,
    pub needed: u32,
    pub tested: u32,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeployReport {
    pub episodes_trained: u32,
    /// Stars earned for training efficiency, if the level has a budget.
    pub stars: Option<u32>,
    pub tasks: Vec<TaskResult>,
    /// The test voyages of the first training task, for display.
    pub voyages: Vec<PlayedEpisode>,
    pub rewards: Vec<RewardSetting>,
}
