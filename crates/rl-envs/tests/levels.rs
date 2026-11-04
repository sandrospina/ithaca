//! Content tests: every level file must parse, validate and compile, and
//! survive random play. Run on every build (plan §14, "Content tests").

use std::path::PathBuf;

use proptest::{prop_assert, prop_assert_eq, proptest};
use rl_core::{Action, Rng, World};
use rl_envs::{LevelDef, load_world};

fn level_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../content/levels")
}

fn level_files() -> Vec<(String, String)> {
    let mut files: Vec<_> = std::fs::read_dir(level_dir())
        .expect("content/levels exists")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "ron"))
        .map(|p| (p.file_name().unwrap().to_string_lossy().into_owned(), std::fs::read_to_string(&p).unwrap()))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no level files found");
    files
}

#[test]
fn every_level_file_compiles() {
    for (name, source) in level_files() {
        let def = LevelDef::parse(&source).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(format!("{}.ron", def.id), name, "level id must match file name");
        load_world(&source).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

#[test]
fn random_play_keeps_invariants_on_every_level() {
    for (name, source) in level_files() {
        let mut world = load_world(&source).unwrap();
        let mut rng = Rng::new(1);
        for episode in 0..200u64 {
            world.reset(episode);
            loop {
                let actions = (0..world.spec().agents.len()).map(|_| Action::Discrete(rng.below(4))).collect();
                let r = world.step(&actions);
                for a in 0..world.spec().agents.len() {
                    let p = world.position(a);
                    assert!(world.grid().in_bounds(p), "{name}: agent left the map at {p:?}");
                    assert!(!world.grid().is_wall(p), "{name}: agent inside a wall at {p:?}");
                }
                assert!(world.step_count() <= world.grid().max_steps, "{name}: exceeded max_steps");
                if r.done() {
                    break;
                }
            }
        }
    }
}

const WALLED: &str = r##"
Level(
  schema: 1, id: "walled", title: "Walled",
  engine: Grid(width: 6, height: 4, moves: Four),
  map: ["S.#...", ".##.#.", "....#G", "#....."],
  legend: { 'S': Spawn(agent: "a"), 'G': Goal(term: "goal_reached") },
  agents: { "a": (controller: Learner, glyph: agent) },
  reward: { "goal_reached": 1.0, "step": -0.1 },
  observation: [ self_position(encoding: Index) ],
  termination: [ GoalReached ],
  truncation: (max_steps: 30),
)
"##;

proptest! {
    #[test]
    fn agent_never_enters_walls_or_leaves_bounds(seed: u64, actions in proptest::collection::vec(0u32..4, 0..60)) {
        let mut world = load_world(WALLED).unwrap();
        world.reset(seed);
        for a in actions {
            let r = world.step(&vec![Action::Discrete(a)]);
            let p = world.position(0);
            prop_assert!(world.grid().in_bounds(p));
            prop_assert!(!world.grid().is_wall(p));
            prop_assert!(r.agents[0].observation.state_index.unwrap() < 24);
            if r.done() { break; }
        }
    }

    #[test]
    fn same_seed_and_actions_give_same_trajectory(seed: u64, actions in proptest::collection::vec(0u32..4, 0..60)) {
        let run = || {
            let mut world = load_world(WALLED).unwrap();
            let mut out = vec![world.reset(seed)[0].clone()];
            for a in &actions {
                let r = world.step(&vec![Action::Discrete(*a)]);
                out.push(r.agents[0].observation.clone());
            }
            out
        };
        prop_assert_eq!(run(), run());
    }
}
