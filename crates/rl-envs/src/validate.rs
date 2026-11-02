//! Level validation. Every error names the field it is about, so level
//! authors (and later teachers) get actionable messages.

use std::fmt;

use crate::catalogue::RewardTerm;
use crate::level::{EngineDef, GoalDef, LegendEntry, LevelDef, SCHEMA_VERSION, TerminationDef};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{field}: {message}")]
pub struct LevelError {
    pub field: String,
    pub message: String,
}

impl LevelError {
    pub fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

/// All problems found in a level file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LevelErrors(pub Vec<LevelError>);

impl fmt::Display for LevelErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, e) in self.0.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{e}")?;
        }
        Ok(())
    }
}

impl std::error::Error for LevelErrors {}

pub const MAX_GRID_SIDE: u32 = 64;

/// What a map character means after applying built-ins and the legend.
pub(crate) fn resolve_char<'a>(def: &'a LevelDef, c: char) -> Option<CellMeaning<'a>> {
    match def.legend.get(&c) {
        Some(LegendEntry::Spawn { agent }) => Some(CellMeaning::Spawn(agent)),
        Some(LegendEntry::Goal { term }) => Some(CellMeaning::Goal(term)),
        Some(LegendEntry::Wall) => Some(CellMeaning::Wall),
        Some(LegendEntry::Open) => Some(CellMeaning::Open),
        None => match c {
            '#' => Some(CellMeaning::Wall),
            '.' => Some(CellMeaning::Open),
            _ => None,
        },
    }
}

pub(crate) enum CellMeaning<'a> {
    Open,
    Wall,
    Spawn(&'a str),
    Goal(&'a str),
}

/// Checks a level definition. Returns every problem found, not just the first.
pub fn validate(def: &LevelDef) -> Result<(), LevelErrors> {
    let mut errors = Vec::new();
    let mut err = |field: &str, message: String| errors.push(LevelError::new(field, message));

    if def.schema != SCHEMA_VERSION {
        err(
            "schema",
            format!("expected {SCHEMA_VERSION}, found {}", def.schema),
        );
    }
    if def.id.trim().is_empty() {
        err("id", "must not be empty".into());
    }

    let EngineDef::Grid { width, height, .. } = def.engine;
    if !(1..=MAX_GRID_SIDE).contains(&width) || !(1..=MAX_GRID_SIDE).contains(&height) {
        err(
            "engine",
            format!("grid size must be 1..={MAX_GRID_SIDE} on each side, found {width}×{height}"),
        );
    }
    if def.map.len() != height as usize {
        err(
            "map",
            format!("expected {height} rows, found {}", def.map.len()),
        );
    }

    let mut spawn_counts = vec![0usize; def.agents.len()];
    let mut goal_count = 0;
    for (y, row) in def.map.iter().enumerate() {
        let len = row.chars().count();
        if len != width as usize {
            err("map", format!("row {y} has {len} cells, expected {width}"));
        }
        for (x, c) in row.chars().enumerate() {
            match resolve_char(def, c) {
                None => err(
                    "map",
                    format!("unknown character '{c}' at row {y}, column {x}; add it to `legend`"),
                ),
                Some(CellMeaning::Spawn(agent)) => match def.agents.get_index_of(agent) {
                    Some(i) => spawn_counts[i] += 1,
                    None => err(
                        "legend",
                        format!("spawn '{c}' refers to unknown agent \"{agent}\""),
                    ),
                },
                Some(CellMeaning::Goal(_)) => goal_count += 1,
                Some(CellMeaning::Wall | CellMeaning::Open) => {}
            }
        }
    }
    for (i, name) in def.agents.keys().enumerate() {
        if spawn_counts[i] != 1 {
            err(
                "map",
                format!(
                    "agent \"{name}\" needs exactly one spawn cell, found {}",
                    spawn_counts[i]
                ),
            );
        }
    }
    if def.agents.is_empty() {
        err("agents", "at least one agent is required".into());
    }

    for (c, entry) in &def.legend {
        if let LegendEntry::Goal { term } = entry
            && RewardTerm::from_name(term).is_none()
        {
            err(
                "legend",
                format!(
                    "goal '{c}' uses unknown reward term \"{term}\"; known terms: {}",
                    RewardTerm::names().join(", ")
                ),
            );
        }
    }

    for (name, weight) in &def.reward {
        if RewardTerm::from_name(name).is_none() {
            err(
                "reward",
                format!(
                    "unknown reward term \"{name}\"; known terms: {}",
                    RewardTerm::names().join(", ")
                ),
            );
        }
        if !weight.is_finite() {
            err(
                "reward",
                format!("weight for \"{name}\" must be a finite number"),
            );
        }
    }

    if def.observation.is_empty() {
        err("observation", "at least one feature is required".into());
    }
    if def.termination.contains(&TerminationDef::GoalReached) && goal_count == 0 {
        err(
            "termination",
            "GoalReached needs at least one goal cell in the map".into(),
        );
    }
    if def.truncation.max_steps == 0 {
        err("truncation", "max_steps must be at least 1".into());
    }

    if let Some(edit) = &def.player_editable.reward {
        for term in &edit.terms {
            if !def.reward.contains_key(term) {
                err(
                    "player_editable.reward",
                    format!("term \"{term}\" is not part of this level's reward"),
                );
            }
        }
        if edit.range.0 > edit.range.1 {
            err(
                "player_editable.reward",
                "range minimum is above its maximum".into(),
            );
        }
    }
    if let Some(h) = &def.player_editable.hyperparameters {
        for (name, bounds) in [
            ("alpha", h.alpha),
            ("gamma", h.gamma),
            ("epsilon", h.epsilon),
        ] {
            if let Some((lo, hi)) = bounds
                && !(0.0..=1.0).contains(&lo) | !(0.0..=1.0).contains(&hi) | (lo > hi)
            {
                err(
                    "player_editable.hyperparameters",
                    format!("{name} bounds must satisfy 0 ≤ min ≤ max ≤ 1"),
                );
            }
        }
    }
    if let Some(c) = &def.challenge {
        if c.objective.trim().is_empty() {
            err("challenge", "objective must not be empty".into());
        }
        match c.goal {
            GoalDef::SuccessRate {
                threshold,
                eval_episodes,
                ..
            } => {
                if !(0.0..=1.0).contains(&threshold) {
                    err("challenge", "threshold must be in 0..=1".into());
                }
                if eval_episodes == 0 {
                    err("challenge", "eval_episodes must be at least 1".into());
                }
            }
            GoalDef::BestReturn => {
                if def.agents.len() != 1 {
                    err("challenge", "BestReturn needs exactly one agent".into());
                }
                if !def.termination.contains(&TerminationDef::GoalReached) {
                    err(
                        "challenge",
                        "BestReturn needs the GoalReached termination".into(),
                    );
                }
                // The shortest path is the best path only if moving costs something.
                if def
                    .reward
                    .get(RewardTerm::Step.name())
                    .is_none_or(|w| *w > 0.0)
                {
                    err(
                        "challenge",
                        "BestReturn needs a step reward of zero or less".into(),
                    );
                }
            }
        }
    }
    if let Some(b) = &def.budget {
        if b.episodes == 0 {
            err("budget", "episodes must be at least 1".into());
        }
        if b.stars.windows(2).any(|w| w[0] < w[1])
            || b.stars.first().is_some_and(|s| *s > b.episodes)
        {
            err(
                "budget",
                "star thresholds must be descending and not exceed the episode budget".into(),
            );
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(LevelErrors(errors))
    }
}
