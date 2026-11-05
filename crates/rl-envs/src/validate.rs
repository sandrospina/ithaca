//! Level validation. Every error names the field it is about, so level
//! authors (and later teachers) get actionable messages.

use std::fmt;

use crate::behaviours::BehaviourDef;
use crate::catalogue::RewardTerm;
use crate::level::{
    ControllerDef, EngineDef, FeatureDef, GoalDef, LegendEntry, LevelDef, QuestionDef, SCHEMA_VERSION, TaskDef,
    TerminationDef,
};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{field}: {message}")]
pub struct LevelError {
    pub field: String,
    pub message: String,
}

impl LevelError {
    pub fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
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
/// Largest grid a scripted chaser may hunt on: its move table grows with the square of the cell count.
pub const MAX_CHASE_CELLS: u32 = 32 * 32;
/// Largest batch of episodes a player can run at once.
pub const MAX_BATCH_EPISODES: u32 = 10_000;

/// What a map character means after applying built-ins and the legend.
pub(crate) fn resolve_char<'a>(def: &'a LevelDef, c: char) -> Option<CellMeaning<'a>> {
    match def.legend.get(&c) {
        Some(LegendEntry::Spawn { agent }) => Some(CellMeaning::Spawn(agent)),
        Some(LegendEntry::Goal { term }) => Some(CellMeaning::Goal(term)),
        Some(LegendEntry::Wall) => Some(CellMeaning::Wall),
        Some(LegendEntry::Open) => Some(CellMeaning::Open),
        Some(LegendEntry::Hazard { lethal }) => Some(CellMeaning::Hazard { lethal: *lethal }),
        Some(LegendEntry::Lure) => Some(CellMeaning::Lure),
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
    Hazard { lethal: bool },
    Lure,
}

/// Problems with a success-rate goal's numbers.
fn rate_problems(threshold: f32, eval_episodes: u32) -> Vec<&'static str> {
    let mut problems = Vec::new();
    if !(0.0..=1.0).contains(&threshold) {
        problems.push("threshold must be in 0..=1");
    }
    if eval_episodes == 0 {
        problems.push("eval_episodes must be at least 1");
    }
    problems
}

/// Checks a level definition. Returns every problem found, not just the first.
pub fn validate(def: &LevelDef) -> Result<(), LevelErrors> {
    let mut errors = Vec::new();
    let mut err = |field: &str, message: String| errors.push(LevelError::new(field, message));

    if def.schema != SCHEMA_VERSION {
        err("schema", format!("expected {SCHEMA_VERSION}, found {}", def.schema));
    }
    if def.id.trim().is_empty() {
        err("id", "must not be empty".into());
    }

    let EngineDef::Grid { width, height, .. } = def.engine;
    if !(1..=MAX_GRID_SIDE).contains(&width) || !(1..=MAX_GRID_SIDE).contains(&height) {
        err("engine", format!("grid size must be 1..={MAX_GRID_SIDE} on each side, found {width}×{height}"));
    }
    if def.map.len() != height as usize {
        err("map", format!("expected {height} rows, found {}", def.map.len()));
    }

    let mut spawn_counts = vec![0usize; def.agents.len()];
    let mut goal_count = 0;
    let mut lethal_count = 0;
    let mut special_count = 0;
    for (y, row) in def.map.iter().enumerate() {
        let len = row.chars().count();
        if len != width as usize {
            err("map", format!("row {y} has {len} cells, expected {width}"));
        }
        for (x, c) in row.chars().enumerate() {
            match resolve_char(def, c) {
                None => err("map", format!("unknown character '{c}' at row {y}, column {x}; add it to `legend`")),
                Some(CellMeaning::Spawn(agent)) => match def.agents.get_index_of(agent) {
                    Some(i) => spawn_counts[i] += 1,
                    None => err("legend", format!("spawn '{c}' refers to unknown agent \"{agent}\"")),
                },
                Some(CellMeaning::Goal(_)) => goal_count += 1,
                Some(CellMeaning::Hazard { lethal: true }) => lethal_count += 1,
                Some(CellMeaning::Hazard { lethal: false } | CellMeaning::Lure) => special_count += 1,
                Some(CellMeaning::Wall | CellMeaning::Open) => {}
            }
        }
    }
    for (i, (name, agent)) in def.agents.iter().enumerate() {
        // A learner starts in one place; a scripted agent may have several
        // spawn cells, one picked per episode from its seed.
        let ok = match agent.controller {
            ControllerDef::Learner => spawn_counts[i] == 1,
            ControllerDef::Scripted { .. } => spawn_counts[i] >= 1,
        };
        if !ok {
            err("map", format!("agent \"{name}\" has {} spawn cells", spawn_counts[i]));
        }
    }
    if def.agents.is_empty() {
        err("agents", "at least one agent is required".into());
    }
    let learners = def.agents.values().take_while(|a| a.controller == ControllerDef::Learner).count();
    if learners == 0 {
        err("agents", "the first agent must be a Learner".into());
    }
    if def.agents.values().skip(learners).any(|a| a.controller == ControllerDef::Learner) {
        err("agents", "learners come before scripted agents".into());
    }
    let scripted: Vec<&str> = def.agents.keys().skip(learners).map(String::as_str).collect();
    for (name, agent) in def.agents.iter().skip(learners) {
        let ControllerDef::Scripted { behaviour: BehaviourDef::ChaseShortestPath { every_n_steps } } =
            &agent.controller
        else {
            continue;
        };
        if *every_n_steps == 0 {
            err("agents", format!("\"{name}\": every_n_steps must be at least 1"));
        }
        if width * height > MAX_CHASE_CELLS {
            err("agents", format!("\"{name}\": chase_shortest_path needs a grid of at most {MAX_CHASE_CELLS} cells"));
        }
    }
    if def.termination.contains(&TerminationDef::CaughtByOpponent) && scripted.is_empty() {
        err("termination", "CaughtByOpponent needs a scripted agent to do the catching".into());
    }
    let feature_problem = |f: &FeatureDef| match f {
        FeatureDef::AgentPosition { agent } | FeatureDef::AgentDirection { agent }
            if !scripted.contains(&agent.as_str()) =>
        {
            Some(format!("{} refers to \"{agent}\", which is not a scripted agent", f.name()))
        }
        _ => None,
    };
    for problem in def.observation.iter().filter_map(feature_problem) {
        err("observation", problem);
    }
    if let Some(edit) = &def.player_editable.observation {
        for problem in edit.allowed.iter().filter_map(feature_problem) {
            err("player_editable.observation", problem);
        }
        if !def.observation.iter().all(|f| edit.allowed.contains(f)) {
            err("player_editable.observation", "the default observation must be among the allowed features".into());
        }
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
            err("reward", format!("unknown reward term \"{name}\"; known terms: {}", RewardTerm::names().join(", ")));
        }
        if !weight.is_finite() {
            err("reward", format!("weight for \"{name}\" must be a finite number"));
        }
    }

    if def.observation.is_empty() {
        err("observation", "at least one feature is required".into());
    }
    if def.termination.contains(&TerminationDef::GoalReached) && goal_count == 0 {
        err("termination", "GoalReached needs at least one goal cell in the map".into());
    }
    if lethal_count > 0 && !def.termination.contains(&TerminationDef::Death) {
        err("termination", "lethal cells need the Death termination".into());
    }
    for name in def.reward_labels.keys() {
        if RewardTerm::from_name(name).is_none() {
            err("reward_labels", format!("unknown reward term \"{name}\""));
        }
    }
    if def.truncation.max_steps == 0 {
        err("truncation", "max_steps must be at least 1".into());
    }

    if let Some(edit) = &def.player_editable.reward {
        for term in &edit.terms {
            if !def.reward.contains_key(term) {
                err("player_editable.reward", format!("term \"{term}\" is not part of this level's reward"));
            }
        }
        if edit.range.0 > edit.range.1 {
            err("player_editable.reward", "range minimum is above its maximum".into());
        }
    }
    if let Some(h) = &def.player_editable.hyperparameters {
        for (name, bounds) in [("alpha", h.alpha), ("gamma", h.gamma), ("epsilon", h.epsilon)] {
            if let Some((lo, hi)) = bounds
                && !(0.0..=1.0).contains(&lo) | !(0.0..=1.0).contains(&hi) | (lo > hi)
            {
                err("player_editable.hyperparameters", format!("{name} bounds must satisfy 0 ≤ min ≤ max ≤ 1"));
            }
        }
    }
    if let Some(c) = &def.challenge {
        if c.objective.trim().is_empty() {
            err("challenge", "objective must not be empty".into());
        }
        match &c.goal {
            GoalDef::SuccessRate { threshold, eval_episodes, .. } => {
                for problem in rate_problems(*threshold, *eval_episodes) {
                    err("challenge", problem.into());
                }
            }
            GoalDef::BestReturn => {
                if lethal_count + special_count > 0 {
                    err("challenge", "BestReturn is only defined for maps without hazards or lures".into());
                }
                if def.agents.len() != 1 {
                    err("challenge", "BestReturn needs exactly one agent".into());
                }
                if !def.termination.contains(&TerminationDef::GoalReached) {
                    err("challenge", "BestReturn needs the GoalReached termination".into());
                }
                // The shortest path is the best path only if moving costs something.
                if def.reward.get(RewardTerm::Step.name()).is_none_or(|w| *w > 0.0) {
                    err("challenge", "BestReturn needs a step reward of zero or less".into());
                }
            }
            GoalDef::Tasks(tasks) => {
                if tasks.is_empty() {
                    err("challenge", "Tasks needs at least one task".into());
                }
                for (i, task) in tasks.iter().enumerate() {
                    match task {
                        TaskDef::BatchRun { min_episodes } if !(1..=MAX_BATCH_EPISODES).contains(min_episodes) => {
                            err("challenge", format!("task {i}: min_episodes must be 1..={MAX_BATCH_EPISODES}"));
                        }
                        TaskDef::CompareObservations { agent, min_gain } => {
                            if !scripted.contains(&agent.as_str()) {
                                err("challenge", format!("task {i}: \"{agent}\" is not a scripted agent"));
                            }
                            if *min_gain == 0 {
                                err("challenge", format!("task {i}: min_gain must be at least 1"));
                            }
                            if def.player_editable.observation.is_none() {
                                err(
                                    "challenge",
                                    format!("task {i}: comparing observations needs an editable observation"),
                                );
                            }
                        }
                        TaskDef::TrainAgent { threshold, eval_episodes, .. } => {
                            for problem in rate_problems(*threshold, *eval_episodes) {
                                err("challenge", format!("task {i}: {problem}"));
                            }
                        }
                        TaskDef::Quiz { questions } => {
                            if questions.is_empty() {
                                err("challenge", format!("task {i}: a quiz needs at least one question"));
                            }
                            for (q, question) in questions.iter().enumerate() {
                                if question.prompt().trim().is_empty() || question.explain().trim().is_empty() {
                                    err(
                                        "challenge",
                                        format!("task {i}, question {q}: prompt and explain are required"),
                                    );
                                }
                                match question {
                                    QuestionDef::StatBand { bands, .. } => {
                                        let ascending = bands.windows(2).all(|w| w[0] < w[1]);
                                        let in_unit = bands.iter().all(|b| (0.0..=1.0).contains(b));
                                        if bands.len() < 3 || !ascending || !in_unit {
                                            err(
                                                "challenge",
                                                format!(
                                                    "task {i}, question {q}: bands must be at least three ascending values in 0..=1"
                                                ),
                                            );
                                        }
                                    }
                                    QuestionDef::Choice { options, answer, .. } => {
                                        if options.len() < 2 || *answer as usize >= options.len() {
                                            err(
                                                "challenge",
                                                format!(
                                                    "task {i}, question {q}: needs two or more options and a valid answer index"
                                                ),
                                            );
                                        }
                                    }
                                    QuestionDef::BestSeed { .. } => {}
                                }
                            }
                        }
                        _ => {}
                    }
                }
                let needs_batch = tasks.iter().any(|t| match t {
                    TaskDef::ReplayBestSeed => true,
                    TaskDef::Quiz { questions } => questions.iter().any(|q| !matches!(q, QuestionDef::Choice { .. })),
                    _ => false,
                });
                if needs_batch && !tasks.iter().any(|t| matches!(t, TaskDef::BatchRun { .. })) {
                    err("challenge", "ReplayBestSeed and questions about a batch need a BatchRun task".into());
                }
            }
        }
    }
    if let Some(b) = &def.budget {
        if b.episodes == 0 {
            err("budget", "episodes must be at least 1".into());
        }
        if b.stars.windows(2).any(|w| w[0] < w[1]) || b.stars.first().is_some_and(|s| *s > b.episodes) {
            err("budget", "star thresholds must be descending and not exceed the episode budget".into());
        }
    }

    if errors.is_empty() { Ok(()) } else { Err(LevelErrors(errors)) }
}
