//! Unit tests of grid mechanics: moves, walls, rewards, termination, truncation, validation.

use rl_core::World;
use rl_envs::{Direction, LevelDef, load_world};

const SMALL: &str = r##"
Level(
  schema: 1, id: "small", title: "Small",
  engine: Grid(width: 3, height: 2, moves: Four),
  map: ["S#G", "..."],
  legend: { 'S': Spawn(agent: "a"), 'G': Goal(term: "goal_reached") },
  agents: { "a": (controller: Learner, glyph: agent) },
  reward: { "goal_reached": 10.0, "step": -1.0 },
  observation: [ self_position(encoding: Index) ],
  termination: [ GoalReached ],
  truncation: (max_steps: 5),
)
"##;

fn step(world: &mut rl_envs::GridWorld, d: Direction) -> rl_core::StepResults {
    world.step(&vec![d.action()])
}

#[test]
fn shortest_path_reaches_goal_with_expected_return() {
    let mut w = load_world(SMALL).unwrap();
    w.reset(0);
    let mut total = 0.0;
    for d in [
        Direction::South,
        Direction::East,
        Direction::East,
        Direction::North,
    ] {
        let r = step(&mut w, d);
        total += r.agents[0].reward;
        if r.done() {
            assert!(r.agents[0].terminated && !r.agents[0].truncated);
        }
    }
    assert_eq!(w.position(0), (2, 0));
    assert!(w.is_done());
    assert_eq!(total, -4.0 + 10.0);
}

#[test]
fn walls_and_edges_block_movement() {
    let mut w = load_world(SMALL).unwrap();
    w.reset(0);
    step(&mut w, Direction::East); // wall
    assert_eq!(w.position(0), (0, 0));
    step(&mut w, Direction::North); // edge
    step(&mut w, Direction::West); // edge
    assert_eq!(w.position(0), (0, 0));
}

#[test]
fn episode_truncates_at_max_steps() {
    let mut w = load_world(SMALL).unwrap();
    w.reset(0);
    let mut last = None;
    for _ in 0..5 {
        last = Some(step(&mut w, Direction::West));
    }
    let r = last.unwrap();
    assert!(r.agents[0].truncated && !r.agents[0].terminated);
    // Stepping after the end is a no-op.
    let after = step(&mut w, Direction::South);
    assert_eq!(after.agents[0].reward, 0.0);
    assert_eq!(w.position(0), (0, 0));
}

#[test]
fn state_index_and_q_table_size() {
    let w = load_world(SMALL).unwrap();
    assert_eq!(w.spec().agents[0].observation.state_count(), Some(6));
}

#[test]
fn validation_reports_every_problem_with_its_field() {
    let bad = SMALL
        .replace(r#""step": -1.0"#, r#""step": -1.0, "lotus": 1.0"#)
        .replace(r#""S#G""#, r#""S#X""#);
    let def = LevelDef::parse(&bad).unwrap();
    let errors = rl_envs::GridWorld::compile(&def).unwrap_err().0;
    let fields: Vec<_> = errors.iter().map(|e| e.field.as_str()).collect();
    assert!(fields.contains(&"reward"), "{errors:?}");
    assert!(fields.contains(&"map"), "{errors:?}");
    assert!(
        errors
            .iter()
            .any(|e| e.message.contains("unknown character 'X'"))
    );
}

#[test]
fn parse_errors_report_position() {
    let err = LevelDef::parse("Level(schema: 1, nonsense: 3)").unwrap_err();
    assert!(err.0[0].field.starts_with("line 1"), "{err}");
}

#[test]
fn best_return_comes_from_the_shortest_path() {
    let w = load_world(SMALL).unwrap();
    // S#G / ... : four moves around the wall, −1 each, +10 at the goal.
    assert_eq!(w.grid().best_return(0), Some((6.0, 4)));
    let loom = load_world(include_str!(
        "../../../content/levels/l00_calypsos_loom.ron"
    ))
    .unwrap();
    assert_eq!(loom.grid().best_return(0), Some((2.0, 8)));
}

#[test]
fn best_return_challenge_rejects_unreachable_goals() {
    let walled_in = SMALL
        .replace("\"S#G\", \"...\"", "\"S#G\", \"##.\"")
        .replace("truncation: (max_steps: 5),", "truncation: (max_steps: 5),\n  challenge: (objective: \"Reach it\", goal: BestReturn),");
    let def = LevelDef::parse(&walled_in).unwrap();
    let errors = rl_envs::GridWorld::compile(&def).unwrap_err();
    assert!(
        errors.to_string().contains("reachable within max_steps"),
        "{errors}"
    );
}

#[test]
fn best_return_challenge_needs_a_step_cost() {
    let free_steps = SMALL.replace(r#""step": -1.0"#, r#""step": 0.5"#).replace(
        "truncation: (max_steps: 5),",
        "truncation: (max_steps: 5),\n  challenge: (objective: \"Reach it\", goal: BestReturn),",
    );
    let def = LevelDef::parse(&free_steps).unwrap();
    let errors = rl_envs::GridWorld::compile(&def).unwrap_err();
    assert!(errors.to_string().contains("step reward"), "{errors}");
}

#[test]
fn task_challenges_are_validated() {
    let with = |goal: &str| {
        let src = SMALL.replace(
            "truncation: (max_steps: 5),",
            &format!("truncation: (max_steps: 5),\n  pilot: Agent,\n  challenge: (objective: \"Measure\", goal: {goal}),"),
        );
        rl_envs::GridWorld::compile(&LevelDef::parse(&src).unwrap())
    };
    assert!(
        with(
            r#"Tasks([BatchRun(min_episodes: 100), ReplayBestSeed, Quiz(questions: [
        StatBand(prompt: "How often?", stat: ShortestRoute, bands: [0.0, 0.5, 1.0], explain: "Because."),
        BestSeed(prompt: "Which seed?", explain: "That one."),
        Choice(prompt: "Why?", options: ["Luck", "Skill"], answer: 0, explain: "Luck."),
    ]), TrainAgent(threshold: 0.9, eval_episodes: 10, eval_seeds: Fixed(1000), success: ShortestRoute)])"#
        )
        .is_ok()
    );

    let errors = with(
        r#"Tasks([ReplayBestSeed, Quiz(questions: [
        StatBand(prompt: "How often?", stat: ReachedGoal, bands: [0.5, 0.2], explain: "x"),
        Choice(prompt: "Why?", options: ["Luck"], answer: 3, explain: "x"),
    ])])"#,
    )
    .unwrap_err()
    .to_string();
    assert!(errors.contains("need a BatchRun task"), "{errors}");
    assert!(errors.contains("bands must be"), "{errors}");
    assert!(errors.contains("valid answer index"), "{errors}");
    assert!(
        with("Tasks([])")
            .unwrap_err()
            .to_string()
            .contains("at least one task")
    );
}
