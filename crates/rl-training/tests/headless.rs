//! Headless simulation: thousands of steps without a renderer, determinism and replay.

use rl_agents::RandomAgent;
use rl_core::{PolicyProvider, World};
use rl_training::{evaluate, replay, run_episode};

const RAFT: &str = include_str!("../../../content/levels/l01_the_raft.ron");

#[test]
fn thousands_of_headless_steps() {
    let mut world = rl_envs::load_world(RAFT).unwrap();
    let mut agent = RandomAgent;
    let mut steps = 0;
    let mut episodes = 0;
    while steps < 50_000 {
        let r = run_episode(&mut world, &mut [&mut agent as &mut dyn PolicyProvider], episodes, episodes);
        steps += r.length;
        episodes += 1;
    }
    assert!(episodes > 100);
}

#[test]
fn same_seeds_give_identical_episodes() {
    let mut world = rl_envs::load_world(RAFT).unwrap();
    let mut a = RandomAgent;
    let r1 = run_episode(&mut world, &mut [&mut a as &mut dyn PolicyProvider], 5, 9);
    let r2 = run_episode(&mut world, &mut [&mut a as &mut dyn PolicyProvider], 5, 9);
    assert_eq!(r1, r2);
    let r3 = run_episode(&mut world, &mut [&mut a as &mut dyn PolicyProvider], 5, 10);
    assert_ne!(r1.actions, r3.actions, "a different policy seed should change random play");
}

#[test]
fn replay_rebuilds_the_episode_from_seed_and_actions() {
    let mut world = rl_envs::load_world(RAFT).unwrap();
    let mut a = RandomAgent;
    let record = run_episode(&mut world, &mut [&mut a as &mut dyn PolicyProvider], 11, 3);
    let steps = replay(&mut world, &record);
    assert_eq!(steps.len() as u32, record.length);
    let total: f32 = steps.iter().map(|s| s.agents[0].reward).sum();
    assert_eq!(total, record.returns[0]);
    assert!(steps.last().unwrap().done());
    assert_eq!(world.step_count(), record.length);
}

#[test]
fn random_agent_is_a_weak_baseline() {
    // The "old way fails" beat of Level 1 relies on random play being poor.
    let mut world = rl_envs::load_world(RAFT).unwrap();
    let mut a = RandomAgent;
    let s = evaluate(&mut world, &mut [&mut a as &mut dyn PolicyProvider], 1000, 200);
    assert!(s.mean_length > 20.0, "random play should wander: {s:?}");
    assert!(s.mean_return < 0.0, "{s:?}");
}
