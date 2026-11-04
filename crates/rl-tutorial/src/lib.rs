//! Tutorials are data: a fixed vocabulary of steps, never code.
//!
//! Steps show text at three depths with UI highlights, ask the player to
//! predict an experiment's result, or run a guided experiment with pinned
//! settings. Every experiment states what it must show, and CI checks it.

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
    /// The CONFIGURE phase: learning settings.
    Configure,
    /// The reward editor inside CONFIGURE.
    Rewards,
    /// The TRAIN phase: training and the learning curve.
    Train,
    /// The DEPLOY phase: testing the trained agent.
    Deploy,
    /// The live Q-learning walkthrough.
    Walkthrough,
    /// The Q-values of the selected cell.
    QValues,
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

/// What a guided experiment measures on the level's test seeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Measure {
    /// Share of test voyages that reach the goal.
    ReachGoal,
    /// Share of test voyages that take the shortest route.
    ShortestRoute,
}

/// A bound an experiment's result must respect, so the lesson it teaches
/// always shows. Rates are fractions of the test voyages.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Expect {
    pub measure: Measure,
    #[serde(default)]
    pub at_least: Option<f32>,
    #[serde(default)]
    pub at_most: Option<f32>,
}

impl Expect {
    pub fn holds(&self, rate: f32) -> bool {
        self.at_least.is_none_or(|lo| rate >= lo - 1e-6) && self.at_most.is_none_or(|hi| rate <= hi + 1e-6)
    }
}

/// A scripted training run: a new agent with pinned settings. Settings left
/// out use the defaults.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentDef {
    #[serde(default)]
    pub alpha: Option<f32>,
    #[serde(default)]
    pub gamma: Option<f32>,
    #[serde(default)]
    pub epsilon: Option<f32>,
    /// Reward weights to set, by term name.
    #[serde(default)]
    pub rewards: std::collections::BTreeMap<String, f32>,
    pub training_seed: u64,
    pub episodes: u32,
    pub expect: Vec<Expect>,
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
    /// The player commits to a prediction, then runs the experiment that answers it.
    Predict {
        beat: Beat,
        title: String,
        text: Text,
        #[serde(default)]
        highlight: Vec<UiElement>,
        question: String,
        options: Vec<String>,
        answer: usize,
        explain: String,
        experiment: ExperimentDef,
    },
    /// A guided experiment the player runs with one button.
    Experiment {
        beat: Beat,
        title: String,
        text: Text,
        #[serde(default)]
        highlight: Vec<UiElement>,
        experiment: ExperimentDef,
    },
}

impl StepDef {
    pub fn beat(&self) -> Beat {
        match self {
            StepDef::Say { beat, .. } | StepDef::Predict { beat, .. } | StepDef::Experiment { beat, .. } => *beat,
        }
    }

    pub fn title(&self) -> &str {
        match self {
            StepDef::Say { title, .. } | StepDef::Predict { title, .. } | StepDef::Experiment { title, .. } => title,
        }
    }

    pub fn text(&self) -> &Text {
        match self {
            StepDef::Say { text, .. } | StepDef::Predict { text, .. } | StepDef::Experiment { text, .. } => text,
        }
    }

    pub fn highlight(&self) -> &[UiElement] {
        match self {
            StepDef::Say { highlight, .. }
            | StepDef::Predict { highlight, .. }
            | StepDef::Experiment { highlight, .. } => highlight,
        }
    }

    pub fn experiment(&self) -> Option<&ExperimentDef> {
        match self {
            StepDef::Say { .. } => None,
            StepDef::Predict { experiment, .. } | StepDef::Experiment { experiment, .. } => Some(experiment),
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
                TutorialError(format!("line {}, column {}: {}", e.span.start.line, e.span.start.col, e.code))
            })?;
        def.validate()?;
        Ok(def)
    }

    pub fn validate(&self) -> Result<(), TutorialError> {
        if self.schema != SCHEMA_VERSION {
            return Err(TutorialError(format!("schema: expected {SCHEMA_VERSION}, found {}", self.schema)));
        }
        if self.steps.is_empty() {
            return Err(TutorialError("steps: at least one step is required".into()));
        }
        for (i, s) in self.steps.iter().enumerate() {
            let err = |m: &str| Err(TutorialError(format!("steps[{i}]: {m}")));
            let text = s.text();
            if s.title().trim().is_empty() || text.story.trim().is_empty() || text.intuition.trim().is_empty() {
                return err("title, story and intuition text are required");
            }
            if let StepDef::Predict { question, options, answer, explain, .. } = s {
                if question.trim().is_empty() || explain.trim().is_empty() {
                    return err("a prediction needs a question and an explanation");
                }
                if options.len() < 2 || *answer >= options.len() {
                    return err("a prediction needs at least two options and an answer among them");
                }
            }
            if let Some(x) = s.experiment() {
                if x.episodes == 0 {
                    return err("an experiment must train for at least one episode");
                }
                let rate_ok = |r: Option<f32>| r.is_none_or(|r| (0.0..=1.0).contains(&r));
                if x.expect.is_empty() || x.expect.iter().any(|e| e.at_least.is_none() && e.at_most.is_none()) {
                    return err("an experiment must state what it shows, with at_least or at_most");
                }
                if x.expect.iter().any(|e| !rate_ok(e.at_least) || !rate_ok(e.at_most)) {
                    return err("expected rates are fractions between 0 and 1");
                }
            }
        }
        let first = self.steps[0].beat();
        if first != Beat::Briefing {
            return Err(TutorialError(format!("steps[0]: a tutorial must open with a Briefing, found {first:?}")));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn with_step(step: &str) -> Result<TutorialDef, TutorialError> {
        TutorialDef::parse(&format!(
            r#"Tutorial(schema: 1, id: "t", level: "l", steps: [
                Say(beat: Briefing, title: "Hi", text: (story: "s", intuition: "i")),
                {step}
            ])"#
        ))
    }

    const EXPERIMENT: &str =
        "(gamma: 0.1, training_seed: 1, episodes: 300, expect: [(measure: ReachGoal, at_least: 0.9)])";

    #[test]
    fn predictions_and_experiments_parse() {
        let def = with_step(&format!(
            r#"Predict(beat: Predict, title: "P", text: (story: "s", intuition: "i"), question: "Q?",
                options: ["a", "b"], answer: 1, explain: "because", experiment: {EXPERIMENT})"#
        ))
        .unwrap();
        let x = def.steps[1].experiment().unwrap();
        assert_eq!((x.gamma, x.alpha, x.episodes), (Some(0.1), None, 300));
        assert!(x.expect[0].holds(0.9) && !x.expect[0].holds(0.89));
        assert!(with_step(&format!(r#"Experiment(beat: GuidedExperiment, title: "E", text: (story: "s", intuition: "i"), experiment: {EXPERIMENT})"#)).is_ok());
    }

    #[test]
    fn bad_predictions_and_experiments_are_rejected() {
        let predict = |options: &str, answer: usize, experiment: &str| {
            with_step(&format!(
                r#"Predict(beat: Predict, title: "P", text: (story: "s", intuition: "i"), question: "Q?",
                    options: {options}, answer: {answer}, explain: "e", experiment: {experiment})"#
            ))
        };
        assert!(predict(r#"["a", "b"]"#, 2, EXPERIMENT).is_err(), "answer out of range");
        assert!(predict(r#"["a"]"#, 0, EXPERIMENT).is_err(), "one option is no choice");
        let no_bound = "(training_seed: 1, episodes: 10, expect: [(measure: ReachGoal)])";
        assert!(predict(r#"["a", "b"]"#, 0, no_bound).is_err(), "an expectation needs a bound");
        let no_episodes = "(training_seed: 1, episodes: 0, expect: [(measure: ReachGoal, at_most: 0.5)])";
        assert!(predict(r#"["a", "b"]"#, 0, no_episodes).is_err());
        let bad_rate = "(training_seed: 1, episodes: 5, expect: [(measure: ReachGoal, at_least: 1.5)])";
        assert!(predict(r#"["a", "b"]"#, 0, bad_rate).is_err());
    }
}
