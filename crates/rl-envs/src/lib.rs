//! Data-driven levels on coded mechanics.
//!
//! A level file (RON) is parsed into a [`LevelDef`], validated, then compiled
//! into a fast runtime world. Level files compose mechanics from fixed
//! catalogues; they never contain code. See `docs/environment-model.md`.

pub mod behaviours;
pub mod catalogue;
pub mod edits;
pub mod grid;
pub mod level;
pub mod validate;

pub use edits::{EditMode, PlayerEdits};
pub use grid::{Direction, GridWorld};
pub use level::{BatchStat, ChallengeDef, EvalSeeds, GoalDef, LevelDef, Pilot, QuestionDef, SuccessDef, TaskDef};
pub use validate::MAX_BATCH_EPISODES;
pub use validate::{LevelError, LevelErrors};

/// FNV-1a hash of a level file's text. Saved policies and experiments record
/// it, so they are never used with a level that has changed since.
pub fn level_hash(source: &str) -> u64 {
    source.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
}

/// Parses, validates and compiles a level file into a playable world.
pub fn load_world(ron_source: &str) -> Result<GridWorld, LevelErrors> {
    let def = LevelDef::parse(ron_source)?;
    GridWorld::compile(&def)
}
