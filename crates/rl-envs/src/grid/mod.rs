//! The Grid engine: discrete cells, integer state, four moves.
//! Used by Levels 0–6.

mod compiled;
mod world;

pub use compiled::CompiledGrid;
pub use world::{Direction, GridWorld};
