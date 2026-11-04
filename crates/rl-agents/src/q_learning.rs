//! Tabular Q-learning: the first learning agent (Level 1 onwards).
//!
//! Deliberately simple and readable; it is also what the tutorials explain
//! line by line.

use rl_core::{Action, ActionMask, Observation, PolicyProvider, Rng};
use serde::{Deserialize, Serialize};

/// Learning settings the player can change within a level's bounds.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct QLearningConfig {
    /// Learning rate α: how far each update moves a Q-value toward its target.
    pub alpha: f32,
    /// Discount γ: how much future rewards count compared with immediate ones.
    pub gamma: f32,
    /// Exploration ε: the chance of a random action during training.
    pub epsilon: f32,
}

impl Default for QLearningConfig {
    fn default() -> Self {
        Self { alpha: 0.5, gamma: 0.95, epsilon: 0.1 }
    }
}

/// One Q-value per (state, action) pair, all starting at zero.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QTable {
    pub states: u32,
    pub actions: u32,
    values: Vec<f32>,
}

impl QTable {
    pub fn new(states: u32, actions: u32) -> Self {
        Self { states, actions, values: vec![0.0; (states * actions) as usize] }
    }

    pub fn get(&self, state: u32, action: u32) -> f32 {
        self.values[(state * self.actions + action) as usize]
    }

    pub fn set(&mut self, state: u32, action: u32, value: f32) {
        self.values[(state * self.actions + action) as usize] = value;
    }

    /// All action values for one state.
    pub fn row(&self, state: u32) -> &[f32] {
        let start = (state * self.actions) as usize;
        &self.values[start..start + self.actions as usize]
    }

    /// Highest value among valid actions.
    pub fn max_value(&self, state: u32, mask: &ActionMask) -> f32 {
        mask.iter().map(|a| self.get(state, a)).fold(f32::NEG_INFINITY, f32::max)
    }

    /// A best valid action. Ties are broken at random, so an untrained table
    /// does not always pick the first action.
    pub fn best_action(&self, state: u32, mask: &ActionMask, rng: &mut Rng) -> u32 {
        let best = self.max_value(state, mask);
        let ties: Vec<u32> = mask.iter().filter(|&a| self.get(state, a) >= best - 1e-6).collect();
        ties[rng.below(ties.len() as u32) as usize]
    }
}

/// What one update did, for the walkthrough and the debugger.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct UpdateTrace {
    pub state: u32,
    pub action: u32,
    pub reward: f32,
    /// `r + γ · max_a' Q(s', a')`, or just `r` when the episode terminated.
    pub target: f32,
    pub td_error: f32,
    pub old_value: f32,
    pub new_value: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QLearner {
    pub config: QLearningConfig,
    pub table: QTable,
}

impl QLearner {
    pub fn new(states: u32, actions: u32, config: QLearningConfig) -> Self {
        Self { config, table: QTable::new(states, actions) }
    }

    /// ε-greedy: explore with probability ε, otherwise take a best action.
    pub fn explore(&self, state: u32, mask: &ActionMask, rng: &mut Rng) -> u32 {
        self.choose(state, mask, rng).0
    }

    /// ε-greedy, also reporting whether the move was a random exploration.
    pub fn choose(&self, state: u32, mask: &ActionMask, rng: &mut Rng) -> (u32, bool) {
        if rng.chance(self.config.epsilon) {
            (mask.nth_valid(rng.below(mask.count_valid())).expect("in range"), true)
        } else {
            (self.table.best_action(state, mask, rng), false)
        }
    }

    /// The Q-learning update for one transition. `next` is the next state and
    /// its action mask, or `None` if the episode terminated. A truncated
    /// episode (out of time) still bootstraps from the next state, because
    /// the world itself did not end.
    pub fn update(&mut self, state: u32, action: u32, reward: f32, next: Option<(u32, &ActionMask)>) -> UpdateTrace {
        let target = match next {
            Some((s2, mask)) => reward + self.config.gamma * self.table.max_value(s2, mask),
            None => reward,
        };
        let old_value = self.table.get(state, action);
        let td_error = target - old_value;
        let new_value = old_value + self.config.alpha * td_error;
        self.table.set(state, action, new_value);
        UpdateTrace { state, action, reward, target, td_error, old_value, new_value }
    }
}

/// The learner's greedy policy: what a trained agent does when deployed.
pub struct GreedyPolicy<'a>(pub &'a QTable);

impl PolicyProvider for GreedyPolicy<'_> {
    fn act(&mut self, observation: &Observation, mask: &ActionMask, rng: &mut Rng) -> Action {
        let state = observation.state_index.expect("Q-learning needs a discrete observation");
        Action::Discrete(self.0.best_action(state, mask, rng))
    }

    fn label(&self) -> &str {
        "trained agent"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_moves_value_toward_target() {
        let mut q = QLearner::new(2, 2, QLearningConfig { alpha: 0.5, gamma: 0.9, epsilon: 0.0 });
        q.table.set(1, 0, 4.0);
        let mask = ActionMask::all(2);
        let t = q.update(0, 1, -1.0, Some((1, &mask)));
        assert!((t.target - (-1.0 + 0.9 * 4.0)).abs() < 1e-6);
        assert!((t.new_value - 1.3).abs() < 1e-6);
        let terminal = q.update(0, 0, 10.0, None);
        assert_eq!((terminal.target, terminal.new_value), (10.0, 5.0));
    }

    #[test]
    fn best_action_respects_mask_and_breaks_ties_randomly() {
        let mut table = QTable::new(1, 4);
        table.set(0, 2, 5.0);
        table.set(0, 3, 5.0);
        let mut mask = ActionMask::all(4);
        let mut rng = Rng::new(1);
        let picks: std::collections::BTreeSet<u32> = (0..50).map(|_| table.best_action(0, &mask, &mut rng)).collect();
        assert_eq!(picks, [2, 3].into());
        mask.set(3, false);
        assert_eq!(table.best_action(0, &mask, &mut rng), 2);
    }
}
