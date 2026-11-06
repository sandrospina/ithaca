//! Level 3, The Cyclops' Cave: the mechanics, and the lessons the tutorial
//! promises. If a change breaks a lesson, these fail.

use rl_agents::QLearningConfig;
use rl_core::{Action, StepEvent, World};
use rl_envs::level::{Encoding, FeatureDef};
use rl_envs::{EditMode, PlayerEdits};
use rl_training::{ExperimentConfig, run_experiment};

const CAVE: &str = include_str!("../../../content/levels/l03_cyclops_cave.ron");
const EAST: u32 = 1;
const SOUTH: u32 = 2;

fn position() -> FeatureDef {
    FeatureDef::SelfPosition { encoding: Encoding::Index }
}

fn direction() -> FeatureDef {
    FeatureDef::AgentDirection { agent: "polyphemus".into() }
}

fn cyclops() -> FeatureDef {
    FeatureDef::AgentPosition { agent: "polyphemus".into() }
}

/// Test escapes of a raft trained with `observation`.
fn escapes(observation: Vec<FeatureDef>, seed: u64, episodes: u32) -> u32 {
    let edits = PlayerEdits { observation: Some(observation), ..Default::default() };
    let c = ExperimentConfig::q_learning("l03_cyclops_cave", CAVE, edits, QLearningConfig::default(), seed, episodes)
        .unwrap();
    run_experiment(&c, CAVE).unwrap().evaluation.reached
}

/// The first seed from 1 whose Polyphemus starts at `spawn`.
fn seed_with(spawn: (i32, i32)) -> u64 {
    let mut world = rl_envs::load_world(CAVE).unwrap();
    (1..)
        .find(|&s| {
            world.reset(s);
            world.positions()[1] == spawn
        })
        .unwrap()
}

#[test]
fn polyphemus_starts_in_a_tunnel_picked_by_the_seed() {
    let mut world = rl_envs::load_world(CAVE).unwrap();
    let spawns: std::collections::BTreeSet<(i32, i32)> = (0..40)
        .map(|s| {
            world.reset(s);
            world.positions()[1]
        })
        .collect();
    assert_eq!(spawns, [(1, 4), (4, 1)].into(), "both tunnels, and nowhere else");
    world.reset(7);
    let first = world.positions();
    world.reset(7);
    assert_eq!(world.positions(), first, "the same seed, the same tunnel");
    assert_eq!(world.spec().agents.len(), 1, "Polyphemus is scripted, not a learner");
}

#[test]
fn polyphemus_chases_and_catches() {
    // Rowing into the north tunnel while he waits there ends badly.
    let mut world = rl_envs::load_world(CAVE).unwrap();
    world.reset(seed_with((4, 1)));
    let r = world.step(&vec![Action::Discrete(EAST)]);
    assert!(!r.done());
    assert_eq!(world.positions()[1], (3, 1), "he closes in");
    let r = world.step(&vec![Action::Discrete(EAST)]);
    assert!(r.agents[0].terminated);
    assert!(r.info.events.iter().any(|e| matches!(e, StepEvent::Caught { agent: 0, by: 1, .. })));
    assert_eq!(r.agents[0].reward, -1.0 - 10.0);

    // The south route escapes him: he follows but never catches up.
    world.reset(seed_with((4, 1)));
    let mut total = 0.0;
    for a in [SOUTH, SOUTH, SOUTH, SOUTH, EAST, EAST, EAST, EAST] {
        total += world.step(&vec![Action::Discrete(a)]).agents[0].reward;
    }
    assert!(world.is_done());
    assert_eq!(total, 2.0, "eight moves and the cave mouth");
}

#[test]
fn the_observation_describes_polyphemus() {
    let mut world = rl_envs::load_world(CAVE).unwrap();
    world
        .apply_edits(
            &PlayerEdits { observation: Some(vec![position(), direction()]), ..Default::default() },
            EditMode::Strict,
        )
        .unwrap();
    assert_eq!(world.spec().agents[0].observation.state_count(), Some(49 * 5));
    world.reset(seed_with((4, 1)));
    let o = world.observe(0);
    assert_eq!(o.features, vec![8.0, 2.0], "the raft at cell 8, Polyphemus to the east");
    world.reset(seed_with((1, 4)));
    assert_eq!(world.observe(0).features, vec![8.0, 3.0], "to the south");
    assert_ne!(world.observe(0).state_index, o.state_index);
}

#[test]
fn only_the_allowed_features_can_be_observed() {
    let mut world = rl_envs::load_world(CAVE).unwrap();
    let edit = |features| PlayerEdits { observation: Some(features), ..Default::default() };
    let odysseus = FeatureDef::AgentPosition { agent: "odysseus".into() };
    assert!(world.apply_edits(&edit(vec![position(), odysseus]), EditMode::Strict).is_err());
    assert!(world.apply_edits(&edit(vec![]), EditMode::Strict).is_err(), "it must see something");
    assert!(world.apply_edits(&edit(vec![position(), position()]), EditMode::Strict).is_err());
    assert!(
        world.apply_edits(&edit(vec![position(), cyclops(), FeatureDef::PreviousAction]), EditMode::Strict).is_ok()
    );
    assert_eq!(world.spec().agents[0].observation.state_count(), Some(49 * 49 * 5));
}

#[test]
fn a_raft_that_sees_only_itself_cannot_pass() {
    // A position-only greedy policy follows one fixed route, which is safe
    // only when Polyphemus waits in the other tunnel: at most 7 of the 10 test seeds.
    for seed in 1..=3 {
        for episodes in [500, 1000, 2000] {
            let n = escapes(vec![position()], seed, episodes);
            assert!(n <= 7, "seed {seed}, {episodes} episodes: {n} escapes");
        }
    }
}

#[test]
fn seeing_which_way_polyphemus_is_wins() {
    for seed in 1..=3 {
        assert_eq!(escapes(vec![position(), direction()], seed, 500), 10, "seed {seed}");
    }
}

#[test]
fn seeing_where_polyphemus_is_wins_too() {
    for seed in 1..=3 {
        assert!(escapes(vec![position(), cyclops()], seed, 2000) >= 8, "seed {seed}");
    }
}

#[test]
fn the_previous_move_does_not_help() {
    for seed in 1..=3 {
        let n = escapes(vec![position(), FeatureDef::PreviousAction], seed, 1000);
        assert!(n <= 7, "seed {seed}: {n} escapes");
    }
}
