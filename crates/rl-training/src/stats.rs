//! Statistics over batches of episodes: what an agent actually achieves
//! across many seeds, not on one lucky voyage.

use serde::{Deserialize, Serialize};

/// Five-number summary plus mean and standard deviation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub mean: f32,
    /// Population standard deviation.
    pub std_dev: f32,
    pub min: f32,
    pub p25: f32,
    pub median: f32,
    pub p75: f32,
    pub max: f32,
}

impl Summary {
    /// Summarises `values`. Returns `None` for an empty slice.
    pub fn of(values: &[f32]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(f32::total_cmp);
        let n = sorted.len() as f64;
        let mean = sorted.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = sorted.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / n;
        Some(Self {
            mean: mean as f32,
            std_dev: var.sqrt() as f32,
            min: sorted[0],
            p25: quantile(&sorted, 0.25),
            median: quantile(&sorted, 0.5),
            p75: quantile(&sorted, 0.75),
            max: sorted[sorted.len() - 1],
        })
    }
}

/// Linear-interpolation quantile of sorted data (the common "type 7" definition).
fn quantile(sorted: &[f32], q: f64) -> f32 {
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = (pos - lo as f64) as f32;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

/// One episode of a batch.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BatchEpisode {
    pub seed: u64,
    pub episode_return: f32,
    pub length: u32,
    pub reached_goal: bool,
}

/// Everything learned from running one agent over consecutive seeds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BatchReport {
    pub agent: String,
    pub first_seed: u64,
    pub episodes: Vec<BatchEpisode>,
    pub returns: Summary,
    pub lengths: Summary,
    /// Share of episodes that reached a goal.
    pub reached_goal_rate: f32,
    /// The best possible return, if known (shortest path to a goal).
    pub best_possible: Option<f32>,
    /// Share of episodes that achieved the best possible return.
    pub shortest_route_rate: Option<f32>,
    /// Index into `episodes` of the highest return (lowest seed on ties).
    pub best: usize,
    /// Index into `episodes` of the lowest return (lowest seed on ties).
    pub worst: usize,
}

/// Mean return of the first `k` episodes, for `k = 1..=n`.
pub fn running_mean(episodes: &[BatchEpisode]) -> Vec<f32> {
    let mut sum = 0.0f64;
    episodes
        .iter()
        .enumerate()
        .map(|(i, e)| {
            sum += f64::from(e.episode_return);
            (sum / (i + 1) as f64) as f32
        })
        .collect()
}

/// Mean return over a sliding window of the last `window` episodes: the
/// usual way to read a learning curve, which a cumulative mean would smooth away.
pub fn moving_mean(episodes: &[BatchEpisode], window: usize) -> Vec<f32> {
    let window = window.max(1);
    let mut sum = 0.0f64;
    episodes
        .iter()
        .enumerate()
        .map(|(i, e)| {
            sum += f64::from(e.episode_return);
            if i >= window {
                sum -= f64::from(episodes[i - window].episode_return);
            }
            (sum / (i + 1).min(window) as f64) as f32
        })
        .collect()
}

/// Tolerance when comparing returns to the best possible return.
const RETURN_EPSILON: f32 = 1e-4;

impl BatchReport {
    /// Builds a report. Returns `None` for an empty batch.
    pub fn new(agent: &str, episodes: Vec<BatchEpisode>, best_possible: Option<f32>) -> Option<Self> {
        let first_seed = episodes.first()?.seed;
        let returns: Vec<f32> = episodes.iter().map(|e| e.episode_return).collect();
        let lengths: Vec<f32> = episodes.iter().map(|e| e.length as f32).collect();
        let n = episodes.len() as f32;
        let best = (0..episodes.len())
            .max_by(|&a, &b| returns[a].total_cmp(&returns[b]).then(episodes[b].seed.cmp(&episodes[a].seed)))?;
        let worst = (0..episodes.len())
            .min_by(|&a, &b| returns[a].total_cmp(&returns[b]).then(episodes[a].seed.cmp(&episodes[b].seed)))?;
        Some(Self {
            agent: agent.to_string(),
            first_seed,
            returns: Summary::of(&returns)?,
            lengths: Summary::of(&lengths)?,
            reached_goal_rate: episodes.iter().filter(|e| e.reached_goal).count() as f32 / n,
            best_possible,
            shortest_route_rate: best_possible.map(|target| {
                episodes.iter().filter(|e| e.reached_goal && e.episode_return >= target - RETURN_EPSILON).count() as f32
                    / n
            }),
            best,
            worst,
            episodes,
        })
    }

    pub fn best_episode(&self) -> &BatchEpisode {
        &self.episodes[self.best]
    }

    pub fn worst_episode(&self) -> &BatchEpisode {
        &self.episodes[self.worst]
    }

    /// Mean return of the first `k` episodes, for `k = 1..=n`. A flat curve
    /// is what "not learning" looks like.
    pub fn running_mean(&self) -> Vec<f32> {
        running_mean(&self.episodes)
    }

    /// Histogram of returns with bins of `width`, aligned to multiples of
    /// `width`. Returns `(bin_start, count)` pairs covering min..=max.
    pub fn histogram(&self, width: f32) -> Vec<(f32, u32)> {
        assert!(width > 0.0);
        let first = (self.returns.min / width).floor();
        let last = (self.returns.max / width).floor();
        let bins = (last - first) as usize + 1;
        let mut counts = vec![0u32; bins];
        for e in &self.episodes {
            let i = ((e.episode_return / width).floor() - first) as usize;
            counts[i.min(bins - 1)] += 1;
        }
        counts.into_iter().enumerate().map(|(i, c)| ((first + i as f32) * width, c)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ep(seed: u64, r: f32, reached: bool) -> BatchEpisode {
        BatchEpisode { seed, episode_return: r, length: 10, reached_goal: reached }
    }

    #[test]
    fn summary_matches_hand_computed_values() {
        let s = Summary::of(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!((s.min, s.max, s.mean), (1.0, 4.0, 2.5));
        assert_eq!((s.p25, s.median, s.p75), (1.75, 2.5, 3.25));
        assert!((s.std_dev - 1.118_034).abs() < 1e-5);
        assert_eq!(Summary::of(&[]), None);
    }

    #[test]
    fn report_finds_best_worst_and_rates() {
        let r = BatchReport::new(
            "random",
            vec![ep(10, -5.0, true), ep(11, 2.0, true), ep(12, 2.0, true), ep(13, -100.0, false)],
            Some(2.0),
        )
        .unwrap();
        assert_eq!(r.best_episode().seed, 11, "ties go to the lowest seed");
        assert_eq!(r.worst_episode().seed, 13);
        assert_eq!(r.reached_goal_rate, 0.75);
        assert_eq!(r.shortest_route_rate, Some(0.5));
        assert_eq!(r.running_mean(), vec![-5.0, -1.5, -1.0 / 3.0, -25.25]);
    }

    #[test]
    fn moving_mean_uses_a_sliding_window() {
        let eps: Vec<_> = [0.0, 2.0, 4.0, 6.0].iter().enumerate().map(|(i, r)| ep(i as u64, *r, true)).collect();
        assert_eq!(moving_mean(&eps, 2), vec![0.0, 1.0, 3.0, 5.0]);
        assert_eq!(running_mean(&eps), vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn histogram_covers_every_episode() {
        let r =
            BatchReport::new("random", vec![ep(0, -12.0, true), ep(1, -3.0, true), ep(2, 2.0, true)], None).unwrap();
        let h = r.histogram(10.0);
        assert_eq!(h, vec![(-20.0, 1), (-10.0, 1), (0.0, 1)]);
        assert_eq!(h.iter().map(|b| b.1).sum::<u32>(), 3);
    }
}
