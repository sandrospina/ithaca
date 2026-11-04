//! Guided-experiment guarantees for Level 2: each reward setting the tutorial
//! suggests must reliably produce its behaviour. If a code change breaks a
//! lesson, this test fails.

use rl_agents::{GreedyPolicy, QLearner, QLearningConfig};
use rl_core::{Rng, World};
use rl_envs::GridWorld;
use rl_training::{EpisodeOutcome, PlayedEpisode, play_episode, train_episode};

const LEVEL: &str = include_str!("../../../content/levels/l02_the_lotus_eaters.ron");

/// Trains with the given reward overrides and returns the greedy policy's test voyages.
fn deploy(rewards: &[(&str, f32)], episodes: u32, seed: u64) -> Vec<PlayedEpisode> {
    let mut world: GridWorld = rl_envs::load_world(LEVEL).unwrap();
    for (name, weight) in rewards {
        world.set_reward_weight(rl_envs::catalogue::RewardTerm::from_name(name).unwrap(), *weight);
    }
    let states = world.spec().agents[0].observation.state_count().unwrap() as u32;
    let mut learner = QLearner::new(states, 4, QLearningConfig::default());
    let mut rng = Rng::new(seed);
    for e in 0..episodes {
        train_episode(&mut world, &mut learner, u64::from(e), &mut rng);
    }
    let mut eval_rng = Rng::new(99);
    (2000..2010).map(|s| play_episode(&mut world, &mut GreedyPolicy(&learner.table), s, &mut eval_rng)).collect()
}

fn count(eps: &[PlayedEpisode], f: impl Fn(&PlayedEpisode) -> bool) -> usize {
    eps.iter().filter(|e| f(e)).count()
}

#[test]
fn map_routes_are_as_designed() {
    let world = rl_envs::load_world(LEVEL).unwrap();
    assert_eq!(world.grid().shortest_path_to_goal(0).map(|(m, _)| m), Some(13));
}

#[test]
fn default_rewards_reach_the_shore_across_the_rocks() {
    for seed in 1..=8 {
        let eps = deploy(&[], 800, seed);
        assert_eq!(count(&eps, |e| e.outcome == EpisodeOutcome::ReachedGoal), 10, "seed {seed}: {eps:?}");
        assert_eq!(count(&eps, |e| e.hazard_steps > 0), 10, "default rewards should cut across the rocks: {eps:?}");
        assert!(eps.iter().all(|e| e.length == 13), "{eps:?}");
    }
}

#[test]
fn a_rich_lotus_breeds_lotus_eaters() {
    for seed in 1..=8 {
        let eps = deploy(&[("lure", 5.0)], 800, seed);
        let stuck = count(&eps, |e| e.outcome == EpisodeOutcome::OutOfTime && e.lure_steps * 2 >= e.length);
        assert_eq!(stuck, 10, "seed {seed}: {eps:?}");
    }
}

#[test]
fn expensive_moves_and_cheap_death_breed_divers() {
    for seed in 1..=8 {
        let eps = deploy(&[("step", -10.0), ("death", -10.0), ("goal_reached", 10.0)], 800, seed);
        assert_eq!(count(&eps, |e| e.outcome == EpisodeOutcome::Died), 10, "seed {seed}: {eps:?}");
    }
}

#[test]
fn costly_rocks_send_the_raft_the_safe_way() {
    for seed in 1..=8 {
        let eps = deploy(&[("hazard", -30.0)], 800, seed);
        let safe = count(&eps, |e| e.outcome == EpisodeOutcome::ReachedGoal && e.hazard_steps == 0);
        assert_eq!(safe, 10, "seed {seed}: {eps:?}");
        assert!(eps.iter().all(|e| e.length == 17), "{eps:?}");
    }
}
