//! Anything that chooses actions: learners, scripted opponents, trained policies.

use crate::{Action, ActionMask, Observation, Rng};

/// Chooses an action for one agent. Opponents are always policy providers,
/// never special-case game logic.
pub trait PolicyProvider {
    fn act(&mut self, observation: &Observation, mask: &ActionMask, rng: &mut Rng) -> Action;

    /// Short label for UI and logs.
    fn label(&self) -> &str;
}
