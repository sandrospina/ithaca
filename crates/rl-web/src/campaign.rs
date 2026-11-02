//! Campaign progress: which levels and tasks are complete, and the player's
//! best results. Saved in the browser's local storage as JSON.

use std::collections::{BTreeMap, BTreeSet};

use rl_envs::{BatchStat, GoalDef, QuestionDef, TaskDef};
use rl_training::BatchReport;
use serde::{Deserialize, Serialize};

use crate::content::LEVELS;
use crate::protocol::{Outcome, SailedBy, Snapshot};

pub const STORAGE_KEY: &str = "reinforcement-odyssey.progress.v1";

/// Tolerance when comparing returns.
const EPSILON: f32 = 1e-4;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Progress {
    pub completed: BTreeSet<String>,
    /// Best return the player achieved by hand, per level.
    pub best_return: BTreeMap<String, f32>,
    /// Done task and question keys per level: `"2"` is task 2, `"2.1"` is question 1 of task 2.
    #[serde(default)]
    pub tasks: BTreeMap<String, BTreeSet<String>>,
}

/// A player's answer to a question.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Answer {
    /// The index of a multiple-choice option or a statistic band.
    Option(u32),
    /// A seed number.
    Seed(u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnswerOutcome {
    Correct,
    Incorrect,
    /// The question is about the player's batch, and there is none yet.
    NeedsBatch,
}

/// A task's state, for the objective checklist.
#[derive(Clone, Debug, PartialEq)]
pub struct TaskState {
    pub label: String,
    pub done: bool,
    /// False when the task needs a feature that is not built yet.
    pub available: bool,
}

fn task_key(task: usize) -> String {
    task.to_string()
}

fn question_key(task: usize, question: usize) -> String {
    format!("{task}.{question}")
}

/// Formats a share in `[0, 1]` as a percentage without needless decimals.
pub fn percent(share: f32) -> String {
    let p = share * 100.0;
    if (p - p.round()).abs() < 1e-3 {
        format!("{}%", p.round() as i64)
    } else {
        format!("{p:.1}%")
    }
}

/// Answer options for a statistic band question, e.g. `0% to 25%`.
pub fn band_labels(bands: &[f32]) -> Vec<String> {
    bands
        .windows(2)
        .map(|w| format!("{} to {}", percent(w[0]), percent(w[1])))
        .collect()
}

/// The value of a batch statistic.
pub fn batch_stat(report: &BatchReport, stat: BatchStat) -> Option<f32> {
    match stat {
        BatchStat::ReachedGoal => Some(report.reached_goal_rate),
        BatchStat::ShortestRoute => report.shortest_route_rate,
    }
}

/// The correct answer to `question`, computed from the player's batch where needed.
pub fn correct_answer(question: &QuestionDef, report: Option<&BatchReport>) -> Option<Answer> {
    match question {
        QuestionDef::Choice { answer, .. } => Some(Answer::Option(*answer)),
        QuestionDef::BestSeed { .. } => report.map(|r| Answer::Seed(r.best_episode().seed)),
        QuestionDef::StatBand { stat, bands, .. } => {
            let value = batch_stat(report?, *stat)?;
            let last = bands.len() - 2;
            let band = bands
                .windows(2)
                .position(|w| value >= w[0] && value < w[1])
                .unwrap_or(last);
            Some(Answer::Option(band as u32))
        }
    }
}

/// Whether a finished episode meets a `BestReturn` objective.
pub fn objective_met(s: &Snapshot) -> bool {
    let Some(objective) = &s.objective else {
        return false;
    };
    match (&objective.goal, objective.best_route) {
        (GoalDef::BestReturn, Some(best)) => {
            s.sailed_by == SailedBy::Player
                && s.outcome == Outcome::ReachedShore
                && s.episode_return >= best.episode_return - EPSILON
        }
        _ => false,
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

    fn done(&self, level_id: &str, key: &str) -> bool {
        self.tasks.get(level_id).is_some_and(|t| t.contains(key))
    }

    fn mark(&mut self, level_id: &str, key: String) {
        self.tasks
            .entry(level_id.to_string())
            .or_default()
            .insert(key);
    }

    fn task_done(&self, level_id: &str, index: usize, task: &TaskDef) -> bool {
        match task {
            TaskDef::Quiz { questions } => {
                (0..questions.len()).all(|q| self.done(level_id, &question_key(index, q)))
            }
            _ => self.done(level_id, &task_key(index)),
        }
    }

    /// Marks the level complete once every task is done. Returns true the
    /// first time that happens.
    fn refresh(&mut self, level_id: &str, goal: &GoalDef) -> bool {
        let GoalDef::Tasks(tasks) = goal else {
            return false;
        };
        let all = tasks
            .iter()
            .enumerate()
            .all(|(i, t)| self.task_done(level_id, i, t));
        all && self.completed.insert(level_id.to_string())
    }

    /// The checklist shown in the objective strip.
    pub fn task_states(&self, level_id: &str, goal: &GoalDef) -> Vec<TaskState> {
        let GoalDef::Tasks(tasks) = goal else {
            return Vec::new();
        };
        tasks
            .iter()
            .enumerate()
            .map(|(i, task)| {
                let label = match task {
                    TaskDef::BatchRun { min_episodes } => {
                        format!("run {min_episodes}+ random voyages")
                    }
                    TaskDef::ReplayBestSeed => "replay the best seed".into(),
                    TaskDef::Quiz { questions } => {
                        let right = (0..questions.len())
                            .filter(|&q| self.done(level_id, &question_key(i, q)))
                            .count();
                        format!("answer Calypso's questions {right}/{}", questions.len())
                    }
                    TaskDef::TrainAgent {
                        threshold,
                        eval_episodes,
                        success,
                        ..
                    } => {
                        let wins = (threshold * *eval_episodes as f32).ceil() as u32;
                        let what = match success {
                            rl_envs::SuccessDef::ShortestRoute => "takes the shortest route",
                            rl_envs::SuccessDef::ReachGoal => "reaches the shore",
                        };
                        format!(
                            "train a raft that {what} in {wins} of {eval_episodes} test voyages"
                        )
                    }
                };
                TaskState {
                    label,
                    done: self.task_done(level_id, i, task),
                    available: !matches!(task, TaskDef::TrainAgent { .. }),
                }
            })
            .collect()
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

    /// Records a batch run. Returns true if this completed the level.
    pub fn record_batch(&mut self, level_id: &str, goal: &GoalDef, report: &BatchReport) -> bool {
        let GoalDef::Tasks(tasks) = goal else {
            return false;
        };
        for (i, task) in tasks.iter().enumerate() {
            if let TaskDef::BatchRun { min_episodes } = task
                && report.episodes.len() >= *min_episodes as usize
            {
                self.mark(level_id, task_key(i));
            }
        }
        self.refresh(level_id, goal)
    }

    /// Records a finished agent episode. If it replays the best seed of
    /// `report`, the replay task is done. Returns true if this completed the level.
    pub fn record_replay(&mut self, s: &Snapshot, report: &BatchReport) -> bool {
        let Some(objective) = &s.objective else {
            return false;
        };
        let GoalDef::Tasks(tasks) = &objective.goal else {
            return false;
        };
        let best = report.best_episode();
        let replayed = s.sailed_by == SailedBy::RandomAgent
            && s.outcome != Outcome::Sailing
            && s.seed == best.seed
            && s.step == best.length
            && (s.episode_return - best.episode_return).abs() < EPSILON;
        if !replayed {
            return false;
        }
        for (i, task) in tasks.iter().enumerate() {
            if matches!(task, TaskDef::ReplayBestSeed) {
                self.mark(&s.level_id, task_key(i));
            }
        }
        self.refresh(&s.level_id, &objective.goal)
    }

    /// Checks an answer to question `question` of task `task` and records a
    /// correct one. Returns the outcome and whether the level was just completed.
    pub fn answer(
        &mut self,
        level_id: &str,
        goal: &GoalDef,
        (task, question): (usize, usize),
        answer: Answer,
        report: Option<&BatchReport>,
    ) -> (AnswerOutcome, bool) {
        let GoalDef::Tasks(tasks) = goal else {
            return (AnswerOutcome::Incorrect, false);
        };
        let Some(TaskDef::Quiz { questions }) = tasks.get(task) else {
            return (AnswerOutcome::Incorrect, false);
        };
        let Some(q) = questions.get(question) else {
            return (AnswerOutcome::Incorrect, false);
        };
        match correct_answer(q, report) {
            None => (AnswerOutcome::NeedsBatch, false),
            Some(right) if right == answer => {
                self.mark(level_id, question_key(task, question));
                (AnswerOutcome::Correct, self.refresh(level_id, goal))
            }
            Some(_) => (AnswerOutcome::Incorrect, false),
        }
    }

    pub fn question_done(&self, level_id: &str, task: usize, question: usize) -> bool {
        self.done(level_id, &question_key(task, question))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{SimRequest, SimResponse};
    use crate::sim::SimCore;

    fn snap(r: SimResponse) -> Snapshot {
        match r {
            SimResponse::Snapshot(s) => *s,
            other => panic!("expected a snapshot, got {other:?}"),
        }
    }

    fn sail(moves: &[u32]) -> Snapshot {
        let mut sim = SimCore::default();
        let mut last = snap(sim.handle(SimRequest::Load {
            level_id: "l00_calypsos_loom".into(),
            seed: 1,
        }));
        for &action in moves {
            last = snap(sim.handle(SimRequest::Step { action }));
        }
        last
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
        p.mark("l01_the_raft", "0".into());
        assert_eq!(Progress::from_json(&p.to_json()), p);
        assert_eq!(Progress::from_json("{not json"), Progress::default());
        // Progress saved before tasks existed still loads.
        let old = r#"{"completed":["l00_calypsos_loom"],"best_return":{"l00_calypsos_loom":2.0}}"#;
        assert!(Progress::from_json(old).is_complete("l00_calypsos_loom"));
    }

    #[test]
    fn percentages_and_band_labels_read_naturally() {
        assert_eq!(percent(0.25), "25%");
        assert_eq!(percent(0.001), "0.1%");
        assert_eq!(
            band_labels(&[0.0, 0.01, 0.05]),
            vec!["0% to 1%", "1% to 5%"]
        );
    }

    /// Plays Level 1's checklist the way a player would.
    #[test]
    fn level_one_tasks_batch_replay_and_quiz() {
        let mut sim = SimCore::default();
        let s = snap(sim.handle(SimRequest::Load {
            level_id: "l01_the_raft".into(),
            seed: 1,
        }));
        let goal = s.objective.as_ref().unwrap().goal.clone();
        let GoalDef::Tasks(tasks) = &goal else {
            panic!("Level 1 uses tasks")
        };
        let quiz = tasks
            .iter()
            .position(|t| matches!(t, TaskDef::Quiz { .. }))
            .unwrap();
        let TaskDef::Quiz { questions } = &tasks[quiz] else {
            unreachable!()
        };
        let mut p = Progress::default();
        let id = "l01_the_raft";

        // Questions about the batch need a batch.
        let (outcome, _) = p.answer(id, &goal, (quiz, 2), Answer::Seed(1), None);
        assert_eq!(outcome, AnswerOutcome::NeedsBatch);

        // A small batch does not satisfy the task; a large one does.
        let small = match sim.handle(SimRequest::Batch {
            first_seed: 1,
            episodes: 20,
        }) {
            SimResponse::Batch(r) => *r,
            other => panic!("{other:?}"),
        };
        assert!(!p.record_batch(id, &goal, &small));
        assert!(!p.task_states(id, &goal)[0].done);
        let report = match sim.handle(SimRequest::Batch {
            first_seed: 1,
            episodes: 100,
        }) {
            SimResponse::Batch(r) => *r,
            other => panic!("{other:?}"),
        };
        p.record_batch(id, &goal, &report);
        assert!(p.task_states(id, &goal)[0].done);

        // Replaying a different seed does not count; the best seed does.
        let best = *report.best_episode();
        let run = |sim: &mut SimCore, seed| {
            sim.handle(SimRequest::Reset { seed });
            let mut s = snap(sim.handle(SimRequest::AgentStep));
            while s.outcome == Outcome::Sailing {
                s = snap(sim.handle(SimRequest::AgentStep));
            }
            s
        };
        p.record_replay(&run(&mut sim, best.seed + 1), &report);
        assert!(!p.task_states(id, &goal)[1].done);
        p.record_replay(&run(&mut sim, best.seed), &report);
        assert!(p.task_states(id, &goal)[1].done);

        // Wrong answers are not recorded; right ones are.
        let (outcome, _) = p.answer(
            id,
            &goal,
            (quiz, 2),
            Answer::Seed(best.seed + 1),
            Some(&report),
        );
        assert_eq!(outcome, AnswerOutcome::Incorrect);
        for (q, question) in questions.iter().enumerate() {
            let right = correct_answer(question, Some(&report)).unwrap();
            let (outcome, completed) = p.answer(id, &goal, (quiz, q), right, Some(&report));
            assert_eq!(outcome, AnswerOutcome::Correct);
            assert!(!completed, "training is still missing");
        }
        let states = p.task_states(id, &goal);
        assert!(states[..3].iter().all(|t| t.done), "{states:?}");
        assert!(!states[3].done && !states[3].available);
        assert!(
            !p.is_complete(id),
            "Level 1 needs a trained agent to complete"
        );
    }

    #[test]
    fn stat_band_answers_follow_the_batch() {
        let level =
            rl_envs::LevelDef::parse(crate::content::level("l01_the_raft").unwrap().level_source)
                .unwrap();
        let GoalDef::Tasks(tasks) = level.challenge.unwrap().goal else {
            panic!()
        };
        let TaskDef::Quiz { questions } = &tasks[2] else {
            panic!()
        };
        let mut sim = SimCore::default();
        sim.handle(SimRequest::Load {
            level_id: "l01_the_raft".into(),
            seed: 1,
        });
        let SimResponse::Batch(report) = sim.handle(SimRequest::Batch {
            first_seed: 1,
            episodes: 1000,
        }) else {
            panic!()
        };
        // About 1 in 1,000 random voyages takes the shortest route: band "0% to 1%".
        assert_eq!(
            correct_answer(&questions[1], Some(&report)),
            Some(Answer::Option(0))
        );
        let reached = report.reached_goal_rate;
        let Some(Answer::Option(band)) = correct_answer(&questions[0], Some(&report)) else {
            panic!()
        };
        let QuestionDef::StatBand { bands, .. } = &questions[0] else {
            panic!()
        };
        assert!(reached >= bands[band as usize] && reached <= bands[band as usize + 1]);
    }
}
