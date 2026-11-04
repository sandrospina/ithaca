//! Render model: what a renderer needs to draw a world, as plain data.
//!
//! The simulation never knows about rendering; renderers only see this model.
//! Glyphs name visual-style tokens (`docs/visual-style.md`), never colours.

use serde::{Deserialize, Serialize};

/// Visual-style glyph tokens. Each RL concept has exactly one glyph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Glyph {
    Agent,
    Adversary,
    Goal,
    Terrain,
    Hazard,
    Policy,
    /// A tempting reward that is not the goal (e.g. the lotus field).
    Lure,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RenderAgent {
    pub name: String,
    pub position: (f32, f32),
    /// Heading in radians, 0 = east, increasing clockwise (screen coordinates).
    pub heading: f32,
    pub glyph: Glyph,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RenderEvent {
    /// Reward ring at a world position; positive values are drawn in gold.
    Reward {
        position: (f32, f32),
        value: f32,
    },
    Bump {
        position: (f32, f32),
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RenderModel {
    /// World size in world units (cells for the Grid engine).
    pub width: f32,
    pub height: f32,
    /// Grid cells that are walls, if the world is a grid.
    pub walls: Vec<(i32, i32)>,
    pub goals: Vec<(i32, i32)>,
    /// Cells that cost reward while the agent is on them.
    #[serde(default)]
    pub hazards: Vec<(i32, i32)>,
    /// Cells that end the episode.
    #[serde(default)]
    pub lethal: Vec<(i32, i32)>,
    /// Cells that pay reward while the agent is on them.
    #[serde(default)]
    pub lures: Vec<(i32, i32)>,
    pub agents: Vec<RenderAgent>,
    pub events: Vec<RenderEvent>,
}
