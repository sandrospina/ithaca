//! Fixed catalogues of reward terms and observation features.
//!
//! Level files refer to entries by name. Adding a mechanic means adding an
//! entry here, with tests, never adding code to a level file.

/// Named reward terms. A level's reward is a weighted sum of these.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RewardTerm {
    /// Paid when an agent enters a goal cell.
    GoalReached,
    /// Paid every step, usually negative, to encourage short paths.
    Step,
}

impl RewardTerm {
    pub const ALL: [RewardTerm; 2] = [RewardTerm::GoalReached, RewardTerm::Step];
    pub const COUNT: usize = Self::ALL.len();

    pub fn name(self) -> &'static str {
        match self {
            RewardTerm::GoalReached => "goal_reached",
            RewardTerm::Step => "step",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.name() == name)
    }

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn names() -> Vec<&'static str> {
        Self::ALL.iter().map(|t| t.name()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for t in RewardTerm::ALL {
            assert_eq!(RewardTerm::from_name(t.name()), Some(t));
            assert_eq!(RewardTerm::ALL[t.index()], t);
        }
        assert_eq!(RewardTerm::from_name("lotus"), None);
    }
}
