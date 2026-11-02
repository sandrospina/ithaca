//! Data-driven levels on coded mechanics.
//!
//! A level file (RON) is parsed into a [`LevelDef`], validated, then compiled
//! into a fast runtime world. Level files compose mechanics from fixed
//! catalogues; they never contain code. See `docs/environment-model.md`.

pub mod catalogue;
pub mod grid;
pub mod level;
pub mod validate;

pub use grid::{Direction, GridWorld};
pub use level::{
    BatchStat, ChallengeDef, EvalSeeds, GoalDef, LevelDef, Pilot, QuestionDef, SuccessDef, TaskDef,
};
pub use validate::MAX_BATCH_EPISODES;
pub use validate::{LevelError, LevelErrors};

/// Parses, validates and compiles a level file into a playable world.
pub fn load_world(ron_source: &str) -> Result<GridWorld, LevelErrors> {
    let def = LevelDef::parse(ron_source)?;
    GridWorld::compile(&def)
}
