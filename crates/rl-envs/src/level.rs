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
    Grid {
        width: u32,
        height: u32,
        moves: Moves,
    },
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

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum GoalDef {
    /// The pilot must reach a goal with the best possible return. The target
    /// is computed from the map, so it can never drift out of date.
    BestReturn,
    /// A trained agent must succeed in `threshold` of `eval_episodes`
    /// evaluation episodes on fixed seeds.
    SuccessRate {
        threshold: f32,
        eval_episodes: u32,
        eval_seeds: EvalSeeds,
    },
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
