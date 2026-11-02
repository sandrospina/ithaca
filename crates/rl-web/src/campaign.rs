//! Campaign progress: which levels are complete, and the player's best
//! results. Saved in the browser's local storage as JSON.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::content::LEVELS;
use crate::protocol::{ObjectiveGoal, Outcome, SailedBy, Snapshot};

pub const STORAGE_KEY: &str = "reinforcement-odyssey.progress.v1";

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Progress {
    pub completed: BTreeSet<String>,
    /// Best return the player achieved by hand, per level.
    pub best_return: BTreeMap<String, f32>,
}

/// Whether a finished episode meets the level's objective.
pub fn objective_met(s: &Snapshot) -> bool {
    match s.objective.as_ref().map(|o| o.goal) {
        Some(ObjectiveGoal::BestReturn { target, .. }) => {
            s.sailed_by == SailedBy::Player
                && s.outcome == Outcome::ReachedShore
                && s.episode_return >= target - 1e-4
        }
        // Completed by evaluating a trained agent, which arrives in Milestone 1.
        Some(ObjectiveGoal::SuccessRate { .. }) | None => false,
    }
}

impl Progress {
    /// Parses saved progress; anything unreadable starts a fresh campaign.
    pub fn from_json(json: &str) -> Self {
        serde_json::from_str(json).unwrap_or_default()
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("progress always serialises")
    }

    pub fn is_complete(&self, level_id: &str) -> bool {
        self.completed.contains(level_id)
    }

    /// The first level is always open; each later level opens when the one
    /// before it is complete.
    pub fn is_unlocked(&self, index: usize) -> bool {
        index == 0
            || LEVELS
                .get(index - 1)
                .is_some_and(|prev| self.is_complete(prev.id))
    }

    /// The furthest level the player can open: where a returning player resumes.
    pub fn resume_index(&self) -> usize {
        (0..LEVELS.len())
            .rev()
            .find(|&i| self.is_unlocked(i))
            .unwrap_or(0)
    }

    pub fn best(&self, level_id: &str) -> Option<f32> {
        self.best_return.get(level_id).copied()
    }

    /// Records a finished episode sailed by the player. Returns true if this
    /// episode completed the level for the first time.
    pub fn record(&mut self, s: &Snapshot) -> bool {
        if s.sailed_by != SailedBy::Player || s.outcome == Outcome::Sailing {
            return false;
        }
        if s.outcome == Outcome::ReachedShore {
            let best = self
                .best_return
                .entry(s.level_id.clone())
                .or_insert(f32::NEG_INFINITY);
            *best = best.max(s.episode_return);
        }
        objective_met(s) && self.completed.insert(s.level_id.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{SimRequest, SimResponse};
    use crate::sim::SimCore;

    fn sail(moves: &[u32]) -> Snapshot {
        let mut sim = SimCore::default();
        let mut last = sim.handle(SimRequest::Load {
            level_id: "l00_calypsos_loom".into(),
            seed: 1,
        });
        for &action in moves {
            last = sim.handle(SimRequest::Step { action });
        }
        match last {
            SimResponse::Snapshot(s) => *s,
            SimResponse::Error(e) => panic!("{e}"),
        }
    }

    #[test]
    fn only_level_zero_is_open_at_the_start() {
        let p = Progress::default();
        assert!(p.is_unlocked(0));
        assert!(!p.is_unlocked(1));
        assert_eq!(p.resume_index(), 0);
    }

    #[test]
    fn a_wandering_voyage_records_a_best_but_does_not_complete() {
        let mut p = Progress::default();
        let detour = sail(&[1, 1, 1, 1, 2, 2, 2, 3, 1, 2]);
        assert_eq!(detour.outcome, Outcome::ReachedShore);
        assert!(!p.record(&detour));
        assert_eq!(p.best("l00_calypsos_loom"), Some(0.0));
        assert!(!p.is_unlocked(1));
    }

    #[test]
    fn the_shortest_voyage_completes_level_zero_and_unlocks_level_one() {
        let mut p = Progress::default();
        let best = sail(&[2, 2, 2, 2, 1, 1, 1, 1]);
        assert!(objective_met(&best));
        assert!(p.record(&best), "first completion is reported");
        assert!(!p.record(&best), "repeat completions are not");
        assert!(p.is_unlocked(1));
        assert_eq!(p.resume_index(), 1);
        assert_eq!(p.best("l00_calypsos_loom"), Some(2.0));
    }

    #[test]
    fn unfinished_episodes_are_ignored() {
        let mut p = Progress::default();
        assert!(!p.record(&sail(&[1, 1])));
        assert!(p.best_return.is_empty());
    }

    #[test]
    fn progress_round_trips_and_survives_corruption() {
        let mut p = Progress::default();
        p.record(&sail(&[2, 2, 2, 2, 1, 1, 1, 1]));
        assert_eq!(Progress::from_json(&p.to_json()), p);
        assert_eq!(Progress::from_json("{not json"), Progress::default());
    }
}
