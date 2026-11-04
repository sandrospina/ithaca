//! The level file format. Authored as RON, exchanged as JSON; both come from
//! these serde types. Bump `SCHEMA_VERSION` on breaking changes.

use indexmap::IndexMap;
use rl_core::Glyph;
use serde::{Deserialize, Serialize};

use crate::validate::{LevelError, LevelErrors};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename = "Level", deny_unknown_fields)]
pub struct LevelDef {
    pub schema: u32,
    pub id: String,
    pub title: String,
    pub engine: EngineDef,
    /// Grid rows, top to bottom. `#` is a wall, `.` is open water; other
    /// characters are defined in `legend`.
    pub map: Vec<String>,
    #[serde(default)]
    pub legend: IndexMap<char, LegendEntry>,
    /// Agents in file order; this order defines `AgentId`s.
    pub agents: IndexMap<String, AgentDef>,
    /// Reward term name → weight.
    pub reward: IndexMap<String, f32>,
    /// Optional level-specific names for reward terms in the reward editor,
    /// e.g. `"lure": "each move in the lotus field"`.
    #[serde(default)]
    pub reward_labels: IndexMap<String, String>,
    pub observation: Vec<FeatureDef>,
    pub termination: Vec<TerminationDef>,
    pub truncation: TruncationDef,
    #[serde(default)]
    pub player_editable: PlayerEditable,
    #[serde(default)]
    pub pilot: Pilot,
    #[serde(default)]
    pub challenge: Option<ChallengeDef>,
    #[serde(default)]
    pub budget: Option<BudgetDef>,
    #[serde(default)]
    pub tutorial: Option<String>,
}

impl LevelDef {
    /// Parses a RON level file. Parse errors report line and column.
    pub fn parse(source: &str) -> Result<Self, LevelErrors> {
        ron::Options::default()
            .with_default_extension(ron::extensions::Extensions::IMPLICIT_SOME)
            .from_str(source)
            .map_err(|e| {
                LevelErrors(vec![LevelError::new(
                    format!("line {}, column {}", e.span.start.line, e.span.start.col),
                    e.code.to_string(),
                )])
            })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EngineDef {
    Grid { width: u32, height: u32, moves: Moves },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Moves {
    /// North, east, south, west.
    Four,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LegendEntry {
    Spawn {
        agent: String,
    },
    /// A goal cell paying the named reward term.
    Goal {
        term: String,
    },
    /// A cell that pays the `hazard` term each step; lethal cells end the
    /// episode and pay the `death` term instead.
    Hazard {
        lethal: bool,
    },
    /// A cell that pays the `lure` term each step the agent stays on it.
    Lure,
    Wall,
    Open,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentDef {
    pub controller: ControllerDef,
    pub glyph: Glyph,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ControllerDef {
    /// Controlled from outside the world: by the player or a learning agent.
    Learner,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureDef {
    SelfPosition { encoding: Encoding },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Encoding {
    /// One number: the cell index. Discrete.
    Index,
    /// Two numbers in `[0, 1]`. Continuous.
    Normalised,
    /// One float per cell, one of them 1. Discrete.
    OneHot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerminationDef {
    GoalReached,
    /// Entering a lethal cell ends the episode.
    Death,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TruncationDef {
    pub max_steps: u32,
}

/// What the player may change, and within which bounds. Everything else is locked.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlayerEditable {
    #[serde(default)]
    pub observation: Option<ObservationEdit>,
    #[serde(default)]
    pub reward: Option<RewardEdit>,
    #[serde(default)]
    pub hyperparameters: Option<HyperparameterBounds>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationEdit {
    pub allowed: Vec<FeatureDef>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RewardEdit {
    pub terms: Vec<String>,
    pub range: (f32, f32),
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HyperparameterBounds {
    #[serde(default)]
    pub alpha: Option<(f32, f32)>,
    #[serde(default)]
    pub gamma: Option<(f32, f32)>,
    #[serde(default)]
    pub epsilon: Option<(f32, f32)>,
}

/// The level's objective: a sentence for the player and a measurable goal.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChallengeDef {
    /// One plain sentence, shown in the objective strip.
    pub objective: String,
    pub goal: GoalDef,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GoalDef {
    /// The pilot must reach a goal with the best possible return. The target
    /// is computed from the map, so it can never drift out of date.
    BestReturn,
    /// A trained agent must succeed in `threshold` of `eval_episodes`
    /// evaluation episodes on fixed seeds.
    SuccessRate { threshold: f32, eval_episodes: u32, eval_seeds: EvalSeeds },
    /// Every task must be done, in any order.
    Tasks(Vec<TaskDef>),
}

/// One task in a level's checklist.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TaskDef {
    /// Run the level's untrained agent over at least `min_episodes` seeds in one batch.
    BatchRun { min_episodes: u32 },
    /// Replay the best seed from the latest batch and see the same voyage again.
    ReplayBestSeed,
    /// Answer every question correctly. Answers come from the player's own batch.
    Quiz { questions: Vec<QuestionDef> },
    /// Train an agent that succeeds in `threshold` of `eval_episodes` test episodes.
    TrainAgent {
        threshold: f32,
        eval_episodes: u32,
        eval_seeds: EvalSeeds,
        success: SuccessDef,
        /// Player-facing checklist label; generated from `success` if absent.
        #[serde(default)]
        label: Option<String>,
    },
}

/// What counts as a successful test episode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuccessDef {
    /// Reaching a goal before the step limit.
    ReachGoal,
    /// Reaching a goal with the best possible return, i.e. by the shortest route.
    ShortestRoute,
    /// Reaching a goal without ever ending a step on a hazard.
    SafeArrival,
    /// A pathology: running out of time with at least half the steps spent on a lure.
    StuckOnLure,
    /// A pathology: ending the episode in a lethal cell.
    Dies,
}

/// A batch statistic a question can ask about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchStat {
    /// Share of episodes that reached a goal.
    ReachedGoal,
    /// Share of episodes that took the shortest route.
    ShortestRoute,
}

/// Question kinds. A fixed vocabulary: questions are data, answers are computed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum QuestionDef {
    /// Which band contains a batch statistic? `bands` are ascending
    /// boundaries in `[0, 1]`; each adjacent pair is one answer option.
    StatBand { prompt: String, stat: BatchStat, bands: Vec<f32>, explain: String },
    /// Which seed gave the best episode in the latest batch?
    BestSeed { prompt: String, explain: String },
    /// A conceptual multiple-choice question with a fixed answer.
    Choice { prompt: String, options: Vec<String>, answer: u32, explain: String },
}

impl QuestionDef {
    pub fn prompt(&self) -> &str {
        match self {
            QuestionDef::StatBand { prompt, .. }
            | QuestionDef::BestSeed { prompt, .. }
            | QuestionDef::Choice { prompt, .. } => prompt,
        }
    }

    pub fn explain(&self) -> &str {
        match self {
            QuestionDef::StatBand { explain, .. }
            | QuestionDef::BestSeed { explain, .. }
            | QuestionDef::Choice { explain, .. } => explain,
        }
    }
}

/// Who steers the learning agent in this level.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pilot {
    /// The player steers by hand (Level 0).
    #[default]
    Player,
    /// An agent steers; the player configures, trains and watches.
    Agent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvalSeeds {
    /// Evaluation episodes use seeds `start, start + 1, …`.
    Fixed(u64),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetDef {
    pub episodes: u32,
    /// Episode thresholds for one, two and three stars, descending.
    pub stars: Vec<u32>,
}
