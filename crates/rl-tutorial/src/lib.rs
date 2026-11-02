//! Tutorials are data: a fixed vocabulary of steps, never code.
//!
//! Milestone 0 supports text steps with UI highlights. Guided experiments,
//! predictions and quizzes are added in Milestone 1 as new `StepDef` variants.

use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename = "Tutorial", deny_unknown_fields)]
pub struct TutorialDef {
    pub schema: u32,
    pub id: String,
    pub level: String,
    pub steps: Vec<StepDef>,
}

/// The seven lesson beats every level follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Beat {
    Briefing,
    OldWayFails,
    Explainer,
    Predict,
    GuidedExperiment,
    Challenge,
    Debrief,
}

impl Beat {
    pub fn label(self) -> &'static str {
        match self {
            Beat::Briefing => "BRIEFING",
            Beat::OldWayFails => "THE OLD WAY FAILS",
            Beat::Explainer => "EXPLAINER",
            Beat::Predict => "PREDICT",
            Beat::GuidedExperiment => "GUIDED EXPERIMENT",
            Beat::Challenge => "CHALLENGE",
            Beat::Debrief => "DEBRIEF",
        }
    }
}

/// How deep an explanation goes. Chosen by the player, remembered per player.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Depth {
    Story,
    #[default]
    Intuition,
    Math,
}

/// UI elements a tutorial step may highlight. A fixed list, so tutorials
/// cannot reach into arbitrary parts of the interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UiElement {
    Arena,
    Agent,
    Goal,
    State,
    Action,
    Reward,
    Return,
    Step,
    Episode,
    Seed,
    Controls,
    /// The objective strip above the arena.
    Objective,
    /// The button that lets an agent sail while the player watches.
    Autopilot,
    /// The ANALYSE phase: batch experiments and statistics.
    Analyse,
    /// The best seed and its replay button.
    BestSeed,
    /// The level's questions.
    Quiz,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Text {
    pub story: String,
    pub intuition: String,
    #[serde(default)]
    pub math: Option<String>,
}

impl Text {
    /// Text at the requested depth. Math falls back to intuition when absent.
    pub fn at(&self, depth: Depth) -> &str {
        match depth {
            Depth::Story => &self.story,
            Depth::Intuition => &self.intuition,
            Depth::Math => self.math.as_deref().unwrap_or(&self.intuition),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum StepDef {
    Say {
        beat: Beat,
        title: String,
        text: Text,
        #[serde(default)]
        highlight: Vec<UiElement>,
    },
}

impl StepDef {
    pub fn beat(&self) -> Beat {
        match self {
            StepDef::Say { beat, .. } => *beat,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct TutorialError(pub String);

impl TutorialDef {
    pub fn parse(source: &str) -> Result<Self, TutorialError> {
        let def: Self = ron::Options::default()
            .with_default_extension(ron::extensions::Extensions::IMPLICIT_SOME)
            .from_str(source)
            .map_err(|e| {
                TutorialError(format!(
                    "line {}, column {}: {}",
                    e.span.start.line, e.span.start.col, e.code
                ))
            })?;
        def.validate()?;
        Ok(def)
    }

    pub fn validate(&self) -> Result<(), TutorialError> {
        if self.schema != SCHEMA_VERSION {
            return Err(TutorialError(format!(
                "schema: expected {SCHEMA_VERSION}, found {}",
                self.schema
            )));
        }
        if self.steps.is_empty() {
            return Err(TutorialError("steps: at least one step is required".into()));
        }
        for (i, s) in self.steps.iter().enumerate() {
            let StepDef::Say { title, text, .. } = s;
            if title.trim().is_empty()
                || text.story.trim().is_empty()
                || text.intuition.trim().is_empty()
            {
                return Err(TutorialError(format!(
                    "steps[{i}]: title, story and intuition text are required"
                )));
            }
        }
        let first = self.steps[0].beat();
        if first != Beat::Briefing {
            return Err(TutorialError(format!(
                "steps[0]: a tutorial must open with a Briefing, found {first:?}"
            )));
        }
        Ok(())
    }
}

/// Progress through a tutorial.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TutorialCursor {
    pub index: usize,
}

impl TutorialCursor {
    pub fn next(&mut self, def: &TutorialDef) -> bool {
        if self.index + 1 < def.steps.len() {
            self.index += 1;
            true
        } else {
            false
        }
    }

    pub fn previous(&mut self) -> bool {
        if self.index > 0 {
            self.index -= 1;
            true
        } else {
            false
        }
    }
}
