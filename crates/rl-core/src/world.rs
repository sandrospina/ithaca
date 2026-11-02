//! The multi-agent world interface. A single-agent level is a world with one
//! externally controlled agent.

use serde::{Deserialize, Serialize};

use crate::{ActionMask, JointAction, Observation, RenderModel, WorldSpec};

pub type AgentId = usize;

/// Something notable that happened during a step. Used by reward terms,
/// tutorials, the HUD and the renderer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum StepEvent {
    Moved {
        agent: AgentId,
        from: (i32, i32),
        to: (i32, i32),
    },
    BumpedWall {
        agent: AgentId,
        at: (i32, i32),
    },
    GoalReached {
        agent: AgentId,
        at: (i32, i32),
    },
    /// A named reward term contributed `value` to `agent`'s reward this step.
    Reward {
        agent: AgentId,
        term: String,
        value: f32,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StepInfo {
    pub events: Vec<StepEvent>,
}

/// Gymnasium-style result for one agent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepResult<O> {
    pub observation: O,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
}

/// Results of one world step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepResults {
    pub agents: Vec<StepResult<Observation>>,
    pub info: StepInfo,
}

impl StepResults {
    /// True if the episode is over for every agent.
    pub fn done(&self) -> bool {
        self.agents.iter().all(|r| r.terminated || r.truncated)
    }
}

pub trait World {
    /// Starts a new episode. The same seed always produces the same episode.
    fn reset(&mut self, seed: u64) -> Vec<Observation>;

    /// Advances one step. `actions` has one entry per agent in `spec().agents`.
    fn step(&mut self, actions: &JointAction) -> StepResults;

    fn action_mask(&self, agent: AgentId) -> ActionMask;

    fn spec(&self) -> &WorldSpec;

    /// Steps taken in the current episode.
    fn step_count(&self) -> u32;

    /// Data-only description for renderers. Never used on the training path.
    fn render_model(&self) -> RenderModel;
}
