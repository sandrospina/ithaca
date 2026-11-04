//! RL regression tests: Q-learning must solve worlds with known answers.

use rl_agents::{GreedyPolicy, QLearner, QLearningConfig};
use rl_core::{Rng, World};
use rl_training::{EpisodeOutcome, play_episode, train_episode};

const RAFT: &str = include_str!("../../../content/levels/l01_the_raft.ron");

fn trained(source: &str, episodes: u32, config: QLearningConfig, seed: u64) -> (rl_envs::GridWorld, QLearner) {
    let mut world = rl_envs::load_world(source).unwrap();
    let states = world.spec().agents[0].observation.state_count().unwrap() as u32;
    let mut learner = QLearner::new(states, 4, config);
    let mut rng = Rng::new(seed);
    for e in 0..episodes {
        train_episode(&mut world, &mut learner, u64::from(e), &mut rng);
    }
    (world, learner)
}

#[test]
fn q_learning_finds_the_shortest_route_on_the_raft() {
    let (mut world, learner) = trained(RAFT, 300, QLearningConfig::default(), 7);
    let mut rng = Rng::new(1);
    for seed in 1000..1010 {
        let ep = play_episode(&mut world, &mut GreedyPolicy(&learner.table), seed, &mut rng);
        assert_eq!(ep.outcome, EpisodeOutcome::ReachedGoal);
        assert_eq!((ep.length, ep.episode_return), (8, 2.0), "seed {seed}");
    }
}

#[test]
fn q_learning_values_approach_the_true_return() {
    // From the start, the optimal return is 2 (eight moves). With γ = 1 the
    // learned value of the best first move converges to it.
    let config = QLearningConfig { alpha: 0.5, gamma: 1.0, epsilon: 0.2 };
    let (_, learner) = trained(RAFT, 2000, config, 3);
    let best = learner.table.row(0).iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!((best - 2.0).abs() < 0.05, "Q(start) = {best}");
}

#[test]
fn a_tiny_discount_makes_the_raft_short_sighted() {
    // With γ = 0.1 the shore is worth almost nothing from the start, so the
    // greedy policy has no reason to head there: the γ lesson of Level 1.
    let config = QLearningConfig { alpha: 0.5, gamma: 0.1, epsilon: 0.1 };
    let (_, learner) = trained(RAFT, 300, config, 7);
    let row = learner.table.row(0);
    let spread =
        row.iter().copied().fold(f32::NEG_INFINITY, f32::max) - row.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(spread < 0.05, "start values barely differ with γ = 0.1: {row:?}");
}

#[test]
fn training_is_reproducible_from_its_seed() {
    let (_, a) = trained(RAFT, 50, QLearningConfig::default(), 11);
    let (_, b) = trained(RAFT, 50, QLearningConfig::default(), 11);
    assert_eq!(a, b);
}
