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
    /// Paid every step the agent ends on a hazard cell (e.g. rocks).
    Hazard,
    /// Paid when the agent enters a lethal cell (e.g. a whirlpool). Ends the episode.
    Death,
    /// Paid every step the agent ends on a lure cell (e.g. the lotus field).
    Lure,
    /// Paid when a scripted opponent catches the agent (e.g. Polyphemus).
    CaughtByOpponent,
}

impl RewardTerm {
    pub const ALL: [RewardTerm; 6] = [
        RewardTerm::GoalReached,
        RewardTerm::Step,
        RewardTerm::Hazard,
        RewardTerm::Death,
        RewardTerm::Lure,
        RewardTerm::CaughtByOpponent,
    ];
    pub const COUNT: usize = Self::ALL.len();

    pub fn name(self) -> &'static str {
        match self {
            RewardTerm::GoalReached => "goal_reached",
            RewardTerm::Step => "step",
            RewardTerm::Hazard => "hazard",
            RewardTerm::Death => "death",
            RewardTerm::Lure => "lure",
            RewardTerm::CaughtByOpponent => "caught_by_opponent",
        }
    }

    /// Plain description for the reward editor, unless the level overrides it.
    pub fn description(self) -> &'static str {
        match self {
            RewardTerm::GoalReached => "reaching the goal",
            RewardTerm::Step => "each move",
            RewardTerm::Hazard => "each move on a hazard",
            RewardTerm::Death => "entering a lethal cell",
            RewardTerm::Lure => "each move inside a lure",
            RewardTerm::CaughtByOpponent => "being caught",
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
