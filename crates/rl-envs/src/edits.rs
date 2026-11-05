//! Player edits to a level, checked against its `player_editable` bounds.
//!
//! The browser and the training servers apply edits through this one
//! function, so a job can never do what the level forbids.

use serde::{Deserialize, Serialize};

use crate::catalogue::RewardTerm;
use crate::grid::GridWorld;
use crate::level::FeatureDef;
use crate::validate::LevelError;

/// What the player changed: reward weights and the agent's observation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlayerEdits {
    /// Reward weights, by term name.
    #[serde(default)]
    pub rewards: Vec<(String, f32)>,
    /// The agent's observation features, replacing the level's default.
    #[serde(default)]
    pub observation: Option<Vec<FeatureDef>>,
}

/// How out-of-bounds values are treated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditMode {
    /// Every problem is an error. For job specifications.
    Strict,
    /// Numbers are clamped into range and edits to locked reward terms are
    /// ignored. For the game's own settings form, which clamps as you type.
    Clamp,
}

impl GridWorld {
    /// Applies `edits` to this world. In [`EditMode::Strict`] nothing is
    /// changed unless every edit is allowed.
    pub fn apply_edits(&mut self, edits: &PlayerEdits, mode: EditMode) -> Result<(), Vec<LevelError>> {
        let mut errors = Vec::new();
        let allowed = self.grid().player_editable.clone();
        let mut weights = Vec::new();
        for (i, (name, weight)) in edits.rewards.iter().enumerate() {
            let field = format!("edits.rewards[{i}]");
            let Some(term) = RewardTerm::from_name(name) else {
                errors.push(LevelError::new(field, format!("unknown reward term \"{name}\"")));
                continue;
            };
            let editable = allowed.reward.as_ref().filter(|r| r.terms.iter().any(|t| t == name));
            let Some(edit) = editable else {
                if mode == EditMode::Strict {
                    errors.push(LevelError::new(field, format!("\"{name}\" is fixed in this level")));
                }
                continue;
            };
            let (lo, hi) = edit.range;
            if !weight.is_finite() {
                if mode == EditMode::Strict {
                    errors.push(LevelError::new(field, format!("\"{name}\" must be a number")));
                }
                continue;
            }
            if mode == EditMode::Strict && !(lo..=hi).contains(weight) {
                errors.push(LevelError::new(field, format!("\"{name}\" is {weight}, outside {lo} to {hi}")));
                continue;
            }
            weights.push((term, weight.clamp(lo, hi)));
        }

        let observation = match &edits.observation {
            None => None,
            // Restating the current observation is not a change.
            Some(features) if *features == self.grid().features => None,
            Some(features) => match check_observation(self, features) {
                Ok(()) => Some(features.clone()),
                Err(e) => {
                    errors.extend(e);
                    None
                }
            },
        };

        if !errors.is_empty() && mode == EditMode::Strict {
            return Err(errors);
        }
        for (term, weight) in weights {
            self.set_reward_weight(term, weight);
        }
        if let Some(features) = observation {
            self.set_observation(features);
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

fn check_observation(world: &GridWorld, features: &[FeatureDef]) -> Result<(), Vec<LevelError>> {
    let mut errors = Vec::new();
    let field = "edits.observation";
    let Some(edit) = &world.grid().player_editable.observation else {
        return Err(vec![LevelError::new(field, "the observation is fixed in this level")]);
    };
    if features.is_empty() {
        errors.push(LevelError::new(field, "the agent must observe at least one feature"));
    }
    for (i, f) in features.iter().enumerate() {
        if !edit.allowed.contains(f) {
            errors
                .push(LevelError::new(format!("{field}[{i}]"), format!("{} is not available in this level", f.name())));
        }
        if features[..i].contains(f) {
            errors.push(LevelError::new(format!("{field}[{i}]"), format!("{} is listed twice", f.name())));
        }
    }
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}
