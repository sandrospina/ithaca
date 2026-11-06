//! Level 4, The Bag of Winds: gusts, the stochastic chaser, and the lessons
//! the tutorial promises. If a change breaks a lesson, these fail.

use rl_agents::QLearningConfig;
use rl_core::{Action, StepEvent, World};
use rl_envs::{EditMode, PlayerEdits};
use rl_training::{ExperimentConfig, run_experiment};

const SEA: &str = include_str!("../../../content/levels/l04_bag_of_winds.ron");
const EAST: u32 = 1;

/// Test arrivals, in the level's 20% wind, of a raft trained in `wind`.
fn arrivals(wind: f32, seed: u64, episodes: u32) -> u32 {
    let edits = PlayerEdits { wind: Some(wind), ..Default::default() };
    let c = ExperimentConfig::q_learning("l04_bag_of_winds", SEA, edits, QLearningConfig::default(), seed, episodes)
        .unwrap();
    run_experiment(&c, SEA).unwrap().evaluation.reached
}

#[test]
fn a_gust_turns_one_move_in_five_aside() {
    let mut world = rl_envs::load_world(SEA).unwrap();
    let (mut straight, mut north, mut south) = (0, 0, 0);
    for seed in 0..4000 {
        world.reset(seed);
        world.place(0, (3, 3));
        world.step(&vec![Action::Discrete(EAST)]);
        match world.position(0) {
            (4, 3) => straight += 1,
            (3, 2) => north += 1,
            (3, 4) => south += 1,
            other => panic!("an east move from (3, 3) cannot end at {other:?}"),
        }
    }
    let share = |n: i32| f64::from(n) / 4000.0;
    assert!((share(straight) - 0.8).abs() < 0.03, "{straight}");
    assert!((share(north) - 0.1).abs() < 0.02 && (share(south) - 0.1).abs() < 0.02, "{north} {south}");
}

#[test]
fn calm_cells_and_calm_water_never_gust() {
    let mut world = rl_envs::load_world(SEA).unwrap();
    for seed in 0..200 {
        world.reset(seed);
        world.place(0, (3, 1));
        let r = world.step(&vec![Action::Discrete(EAST)]);
        assert_eq!(world.position(0), (4, 1), "row 1 has no wind");
        assert!(!r.info.events.iter().any(|e| matches!(e, StepEvent::Gust { .. })));
    }
    world.apply_edits(&PlayerEdits { wind: Some(0.0), ..Default::default() }, EditMode::Strict).unwrap();
    for seed in 0..200 {
        world.reset(seed);
        world.place(0, (3, 3));
        world.step(&vec![Action::Discrete(EAST)]);
        assert_eq!(world.position(0), (4, 3), "a calm sea goes where it is steered");
    }
}

#[test]
fn gusts_replay_exactly_from_the_seed() {
    let mut world = rl_envs::load_world(SEA).unwrap();
    let path = |world: &mut rl_envs::GridWorld, seed| {
        world.reset(seed);
        (0..6).map(|_| world.step(&vec![Action::Discrete(EAST)]).agents[0].reward).collect::<Vec<f32>>()
    };
    for seed in 0..50 {
        assert_eq!(path(&mut world, seed), path(&mut world, seed));
    }
}

#[test]
fn the_wind_can_only_be_set_within_its_range() {
    let mut world = rl_envs::load_world(SEA).unwrap();
    let wind = |w| PlayerEdits { wind: Some(w), ..Default::default() };
    assert!(world.apply_edits(&wind(0.5), EditMode::Strict).is_err());
    assert!(world.apply_edits(&wind(f32::NAN), EditMode::Strict).is_err());
    assert!(world.apply_edits(&wind(0.3), EditMode::Strict).is_ok());
    let mut cave = rl_envs::load_world(include_str!("../../../content/levels/l03_cyclops_cave.ron")).unwrap();
    assert!(cave.apply_edits(&wind(0.1), EditMode::Strict).is_err(), "the cave has no wind to change");
}

#[test]
fn a_stochastic_chaser_varies_but_replays() {
    let level = |p: f32| {
        format!(
            r#"Level(schema: 1, id: "t", title: "t", engine: Grid(width: 5, height: 5, moves: Four),
            map: [".....", ".....", "..C..", ".....", "S...G"],
            legend: {{ 'S': Spawn(agent: "a"), 'C': Spawn(agent: "c"), 'G': Goal(term: "goal_reached") }},
            agents: {{ "a": (controller: Learner, glyph: agent),
                       "c": (controller: Scripted(behaviour: chase_shortest_path(every_n_steps: 1, p_random: {p})), glyph: adversary) }},
            reward: {{ "goal_reached": 1.0 }}, observation: [ self_position(encoding: Index) ],
            termination: [ GoalReached, CaughtByOpponent ], truncation: (max_steps: 20))"#
        )
    };
    let first_move = |source: &str, seed| {
        let mut w = rl_envs::load_world(source).unwrap();
        w.reset(seed);
        w.step(&vec![Action::Discrete(0)]);
        w.positions()[1]
    };
    let steady = level(0.0);
    let shaky = level(1.0);
    let steady_moves: std::collections::BTreeSet<_> = (0..30).map(|s| first_move(&steady, s)).collect();
    let shaky_moves: std::collections::BTreeSet<_> = (0..30).map(|s| first_move(&shaky, s)).collect();
    assert_eq!(steady_moves.len(), 1, "p_random = 0 always takes the shortest path");
    assert!(shaky_moves.len() >= 3, "p_random = 1 wanders: {shaky_moves:?}");
    assert_eq!(first_move(&shaky, 7), first_move(&shaky, 7), "the same seed, the same lurch");
}

#[test]
fn a_raft_trained_in_calm_water_is_lost_in_the_wind() {
    for seed in 1..=5 {
        let n = arrivals(0.0, seed, 1500);
        assert!(n <= 7, "seed {seed}: {n} arrivals");
    }
}

#[test]
fn a_raft_trained_in_the_wind_takes_the_safe_way() {
    for seed in 1..=5 {
        assert_eq!(arrivals(0.2, seed, 800), 10, "seed {seed}");
    }
}
