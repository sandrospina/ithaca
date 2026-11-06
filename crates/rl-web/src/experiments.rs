//! The player's experiment log for one level: every run, kept with its full
//! specification so it can be rerun, compared and replayed.
//!
//! Plain Rust, tested natively. The log is saved in the browser next to the
//! level's raft.

use rl_envs::level::FeatureDef;
use rl_training::{BatchEpisode, EvaluationReport, ExperimentConfig, ExperimentRun};
use serde::{Deserialize, Serialize};

use crate::chart_math::signed;

/// Version of [`ExperimentLog`]; a log of another version is ignored.
pub const LOG_SCHEMA: u32 = 1;
/// Runs kept per level. The oldest goes when a new one arrives.
pub const MAX_LOGGED: usize = 12;
/// Runs shown together in the comparison.
pub const MAX_COMPARED: usize = 4;
/// Episodes averaged by the comparison's learning curves.
pub const COMPARE_WINDOW: usize = 50;

/// One run in the log. Its policy is not kept: rerunning the specification
/// rebuilds it exactly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoggedRun {
    pub id: u32,
    /// Where it came from, e.g. "run 3" or a tutorial step's title.
    pub label: String,
    pub config: ExperimentConfig,
    pub engine: String,
    pub fingerprint: u64,
    pub curve: Vec<BatchEpisode>,
    pub evaluation: EvaluationReport,
    /// Time from request to result, measured by the page.
    pub wall_clock_ms: Option<f64>,
}

impl LoggedRun {
    pub fn training_steps(&self) -> u64 {
        self.curve.iter().map(|e| u64::from(e.length)).sum()
    }

    /// Share of training moves chosen at random.
    pub fn exploration_rate(&self) -> f32 {
        let explored: u64 = self.curve.iter().map(|e| u64::from(e.explored)).sum();
        explored as f32 / self.training_steps().max(1) as f32
    }

    /// The settings in one line: learning settings, edits, episodes and seed.
    pub fn settings(&self) -> String {
        settings_line(&self.config)
    }

    /// Short form of the fingerprint, for display.
    pub fn short_fingerprint(&self) -> String {
        format!("{:016x}", self.fingerprint)[..8].to_string()
    }
}

/// An experiment's settings in one line.
pub fn settings_line(c: &ExperimentConfig) -> String {
    let q = c.algorithm.q_learning();
    let mut parts = vec![format!("α {} · γ {} · ε {}", q.alpha, q.gamma, q.epsilon)];
    parts.extend(c.edits.rewards.iter().map(|(k, v)| format!("{k} {}", signed(f64::from(*v)))));
    if let Some(features) = &c.edits.observation {
        parts.push(observation_line(features));
    }
    if let Some(w) = c.edits.wind {
        parts.push(wind_line(w));
    }
    parts.push(format!("{} episodes · seed {}", c.episodes, c.seed));
    parts.join(" · ")
}

/// An agent id as the player reads it: "polyphemus" becomes "Polyphemus".
pub fn display_name(id: &str) -> String {
    let mut c = id.chars();
    c.next().map(|f| f.to_uppercase().chain(c).collect()).unwrap_or_default()
}

/// Whether an observation includes `agent`, by its position or its direction.
pub fn sees(features: &[FeatureDef], agent: &str) -> bool {
    features.iter().any(|f| match f {
        FeatureDef::AgentPosition { agent: a } | FeatureDef::AgentDirection { agent: a } => a == agent,
        _ => false,
    })
}

/// The training wind, in a few words: "trains in calm water", "trains in 20% wind".
pub fn wind_line(slip: f32) -> String {
    if slip == 0.0 { "trains in calm water".into() } else { format!("trains in {:.0}% wind", slip * 100.0) }
}

/// What an agent observes, in a few words.
pub fn observation_line(features: &[FeatureDef]) -> String {
    format!("sees {}", features.iter().map(FeatureDef::name).collect::<Vec<_>>().join(" + "))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperimentLog {
    pub schema: u32,
    pub level_id: String,
    pub next_id: u32,
    pub runs: Vec<LoggedRun>,
}

impl ExperimentLog {
    pub fn new(level_id: &str) -> Self {
        Self { schema: LOG_SCHEMA, level_id: level_id.to_string(), next_id: 1, runs: Vec::new() }
    }

    /// The browser storage key of a level's log.
    pub fn storage_key(level_id: &str) -> String {
        format!("experiments/{level_id}")
    }

    /// Reads a saved log, if it is for this level and this version.
    pub fn from_json(level_id: &str, json: &str) -> Option<Self> {
        serde_json::from_str::<Self>(json).ok().filter(|l| l.schema == LOG_SCHEMA && l.level_id == level_id)
    }

    /// Adds a run and returns its id. `label` defaults to "run N".
    pub fn add(&mut self, run: &ExperimentRun, label: Option<String>, wall_clock_ms: Option<f64>) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.runs.push(LoggedRun {
            id,
            label: label.unwrap_or_else(|| format!("run {id}")),
            config: run.config.clone(),
            engine: run.engine.clone(),
            fingerprint: run.fingerprint(),
            curve: run.curve.clone(),
            evaluation: run.evaluation.clone(),
            wall_clock_ms,
        });
        if self.runs.len() > MAX_LOGGED {
            self.runs.remove(0);
        }
        id
    }

    pub fn get(&self, id: u32) -> Option<&LoggedRun> {
        self.runs.iter().find(|r| r.id == id)
    }

    pub fn remove(&mut self, id: u32) {
        self.runs.retain(|r| r.id != id);
    }

    /// The largest gain in test arrivals from training in a stronger wind, over
    /// pairs of runs whose settings differ only in the training wind. `level`
    /// is the level's own slip, used by runs that did not change it.
    pub fn wind_gain(&self, level: f32) -> Option<i64> {
        let wind = |r: &LoggedRun| r.config.edits.wind.unwrap_or(level);
        let without_wind = |r: &LoggedRun| {
            let mut c = r.config.clone();
            c.edits.wind = None;
            c
        };
        let mut best = None;
        for a in &self.runs {
            for b in self.runs.iter().filter(|b| wind(b) < wind(a)) {
                if without_wind(a) == without_wind(b) {
                    let gain = i64::from(a.evaluation.reached) - i64::from(b.evaluation.reached);
                    best = Some(best.map_or(gain, |g: i64| g.max(gain)));
                }
            }
        }
        best
    }

    /// The largest gain in test arrivals from observing `agent`, over pairs of
    /// runs whose settings differ only in the observation: one sees the agent,
    /// the other does not. `default` is the level's own observation, used by
    /// runs that did not change it. `None` without such a pair.
    pub fn observation_gain(&self, agent: &str, default: &[FeatureDef]) -> Option<i64> {
        let features = |r: &LoggedRun| r.config.edits.observation.clone().unwrap_or_else(|| default.to_vec());
        let without_observation = |r: &LoggedRun| {
            let mut c = r.config.clone();
            c.edits.observation = None;
            c
        };
        let mut best = None;
        for a in self.runs.iter().filter(|r| sees(&features(r), agent)) {
            for b in self.runs.iter().filter(|r| !sees(&features(r), agent)) {
                if without_observation(a) == without_observation(b) {
                    let gain = i64::from(a.evaluation.reached) - i64::from(b.evaluation.reached);
                    best = Some(best.map_or(gain, |g: i64| g.max(gain)));
                }
            }
        }
        best
    }
}

/// Which runs are compared. Each run holds a fixed slot while it stays
/// selected; the slot picks its colour, dash and letter, so removing one
/// run never repaints the others.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Selection(pub [Option<u32>; MAX_COMPARED]);

impl Selection {
    /// Selects `id` in the first free slot, or frees its slot. Returns false
    /// if every slot is taken.
    pub fn toggle(&mut self, id: u32) -> bool {
        if let Some(i) = self.slot(id) {
            self.0[i] = None;
            return true;
        }
        match self.0.iter().position(Option::is_none) {
            Some(i) => {
                self.0[i] = Some(id);
                true
            }
            None => false,
        }
    }

    pub fn slot(&self, id: u32) -> Option<usize> {
        self.0.iter().position(|&x| x == Some(id))
    }

    pub fn is_full(&self) -> bool {
        self.0.iter().all(Option::is_some)
    }

    /// Selected runs with their slots, in slot order.
    pub fn slots(&self) -> impl Iterator<Item = (usize, u32)> + '_ {
        self.0.iter().enumerate().filter_map(|(i, x)| x.map(|id| (i, id)))
    }

    /// Frees the slots of runs no longer in the log.
    pub fn retain_in(&mut self, log: &ExperimentLog) {
        for x in &mut self.0 {
            if x.is_some_and(|id| log.get(id).is_none()) {
                *x = None;
            }
        }
    }
}

/// Letters naming the comparison slots in the chart, its legend and the table.
pub const SLOT_LETTERS: [&str; MAX_COMPARED] = ["A", "B", "C", "D"];

#[cfg(test)]
mod tests {
    use super::*;
    use rl_agents::QLearningConfig;
    use rl_envs::PlayerEdits;

    fn run(gamma: f32) -> ExperimentRun {
        let source = crate::content::level("l01_the_raft").unwrap().level_source;
        let config = ExperimentConfig::q_learning(
            "l01_the_raft",
            source,
            PlayerEdits::default(),
            QLearningConfig { gamma, ..Default::default() },
            1,
            100,
        )
        .unwrap();
        rl_training::run_experiment(&config, source).unwrap()
    }

    #[test]
    fn the_log_keeps_runs_and_survives_a_round_trip() {
        let mut log = ExperimentLog::new("l01_the_raft");
        let a = log.add(&run(0.95), None, Some(12.0));
        let b = log.add(&run(0.1), Some("A short-sighted raft".into()), None);
        assert_eq!((a, b), (1, 2));
        assert_eq!(log.get(1).unwrap().label, "run 1");
        assert_eq!(log.get(2).unwrap().settings(), "α 0.5 · γ 0.1 · ε 0.1 · 100 episodes · seed 1");
        let json = serde_json::to_string(&log).unwrap();
        assert_eq!(ExperimentLog::from_json("l01_the_raft", &json), Some(log.clone()));
        assert_eq!(ExperimentLog::from_json("l02_the_lotus_eaters", &json), None, "another level's log");
        let r = log.get(1).unwrap();
        assert_eq!(r.curve.len(), 100);
        assert!(r.exploration_rate() > 0.0);
    }

    #[test]
    fn the_log_is_bounded() {
        let mut log = ExperimentLog::new("l01_the_raft");
        let r = run(0.95);
        for _ in 0..MAX_LOGGED + 2 {
            log.add(&r, None, None);
        }
        assert_eq!(log.runs.len(), MAX_LOGGED);
        assert_eq!(log.runs[0].id, 3, "the oldest runs go first");
    }

    #[test]
    fn a_compared_run_keeps_its_slot() {
        let mut s = Selection::default();
        for id in [4, 7, 9] {
            assert!(s.toggle(id));
        }
        assert_eq!(s.slot(9), Some(2));
        s.toggle(7);
        assert_eq!((s.slot(4), s.slot(9)), (Some(0), Some(2)), "removing B does not repaint C");
        assert!(s.toggle(1));
        assert_eq!(s.slot(1), Some(1), "a new run takes the free slot");
        assert!(s.toggle(2));
        assert!(s.is_full());
        assert!(!s.toggle(3), "a fifth run is refused");
    }
}
