//! The random agent: the baseline every level's "old way fails" beat starts from.

use rl_core::{Action, ActionMask, Observation, PolicyProvider, Rng};

#[derive(Clone, Debug, Default)]
pub struct RandomAgent;

impl PolicyProvider for RandomAgent {
    fn act(&mut self, _observation: &Observation, mask: &ActionMask, rng: &mut Rng) -> Action {
        let n = mask.count_valid();
        assert!(n > 0, "no valid actions");
        Action::Discrete(mask.nth_valid(rng.below(n)).expect("n is in range"))
    }

    fn label(&self) -> &str {
        "random"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn respects_mask() {
        let mut mask = ActionMask::all(4);
        mask.set(0, false);
        mask.set(2, false);
        let obs = Observation {
            features: vec![],
            state_index: None,
        };
        let mut rng = Rng::new(3);
        let mut agent = RandomAgent;
        for _ in 0..200 {
            let a = agent.act(&obs, &mask, &mut rng).discrete();
            assert!(a == 1 || a == 3);
        }
    }
}
