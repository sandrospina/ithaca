//! The simulation side of the client: owns the world, tracks the episode,
//! and answers requests with snapshots. Plain Rust, tested natively.

use rl_agents::RandomAgent;
use rl_core::{Action, ActionSpace, PolicyProvider, Rng, StepEvent, World};
use rl_envs::{GoalDef, GridWorld, Pilot};

use crate::content;
use crate::protocol::{
    Baseline, Objective, ObjectiveGoal, Outcome, SailedBy, SimRequest, SimResponse, Snapshot,
};

/// Episodes used to measure the random agent's baseline on a level.
pub const BASELINE_EPISODES: u32 = 100;

#[derive(Default)]
pub struct SimCore {
    level_id: String,
    world: Option<GridWorld>,
    objective: Option<Objective>,
    random_baseline: Option<Baseline>,
    random_agent: RandomAgent,
    policy_rng: Option<Rng>,
    sailed_by: Option<SailedBy>,
    seed: u64,
    episode: u32,
    last_action: Option<String>,
    last_reward: Option<f32>,
    episode_return: f32,
    outcome: Option<Outcome>,
    log: Vec<String>,
}

impl SimCore {
    pub fn handle(&mut self, request: SimRequest) -> SimResponse {
        match request {
            SimRequest::Load { level_id, seed } => {
                if let Err(e) = self.load(&level_id) {
                    return SimResponse::Error(e);
                }
                self.reset(seed);
            }
            SimRequest::Reset { seed } => {
                if self.world.is_none() {
                    return SimResponse::Error("no level loaded".into());
                }
                self.reset(seed);
            }
            SimRequest::Step { action } => {
                let Some(world) = &self.world else {
                    return SimResponse::Error("no level loaded".into());
                };
                if world.grid().pilot == Pilot::Agent {
                    self.log = vec!["steering is off · the raft is the agent's job now".into()];
                } else {
                    self.step(action, SailedBy::Player);
                }
            }
            SimRequest::AgentStep => {
                let Some(world) = &self.world else {
                    return SimResponse::Error("no level loaded".into());
                };
                if world.grid().pilot == Pilot::Player {
                    self.log = vec!["in this level you are the pilot".into()];
                } else {
                    let observation = world.observe(0);
                    let mask = world.action_mask(0);
                    let rng = self.policy_rng.as_mut().expect("set on reset");
                    let action = self.random_agent.act(&observation, &mask, rng).discrete();
                    self.step(action, SailedBy::RandomAgent);
                }
            }
        }
        SimResponse::Snapshot(Box::new(self.snapshot()))
    }

    fn load(&mut self, level_id: &str) -> Result<(), String> {
        let entry =
            content::level(level_id).ok_or_else(|| format!("unknown level \"{level_id}\""))?;
        let world =
            rl_envs::load_world(entry.level_source).map_err(|e| format!("{level_id}: {e}"))?;
        let grid = world.grid();
        self.objective = grid.challenge.as_ref().map(|c| Objective {
            text: c.objective.clone(),
            goal: match c.goal {
                GoalDef::BestReturn => {
                    let (target, moves) = grid.best_return(0).expect("validated at compile time");
                    ObjectiveGoal::BestReturn { target, moves }
                }
                GoalDef::SuccessRate {
                    threshold,
                    eval_episodes,
                    ..
                } => ObjectiveGoal::SuccessRate {
                    threshold,
                    eval_episodes,
                },
            },
        });
        self.random_baseline = (grid.pilot == Pilot::Agent).then(|| {
            let mut probe = world.clone();
            let mut agent = RandomAgent;
            let s = rl_training::evaluate(
                &mut probe,
                &mut [&mut agent as &mut dyn PolicyProvider],
                0,
                BASELINE_EPISODES,
            );
            Baseline {
                episodes: s.episodes,
                success_rate: s.success_rate,
                mean_return: s.mean_return,
                mean_length: s.mean_length,
            }
        });
        self.world = Some(world);
        self.level_id = level_id.to_string();
        self.episode = 0;
        Ok(())
    }

    fn reset(&mut self, seed: u64) {
        let world = self.world.as_mut().expect("checked");
        world.reset(seed);
        self.policy_rng = Some(Rng::new(seed).fork(0x5241_4654)); // "RAFT"
        self.sailed_by = Some(SailedBy::Nobody);
        self.seed = seed;
        self.episode += 1;
        self.last_action = None;
        self.last_reward = None;
        self.episode_return = 0.0;
        self.outcome = Some(Outcome::Sailing);
        self.log = vec![format!("episode {} begins · seed {seed}", self.episode)];
    }

    fn step(&mut self, action: u32, by: SailedBy) {
        let world = self.world.as_mut().expect("checked");
        if world.is_done() {
            self.log = vec!["the episode is over · press R to sail again".into()];
            return;
        }
        let ActionSpace::Discrete { labels, .. } = &world.spec().agents[0].action;
        let label = labels.get(action as usize).cloned();
        let result = world.step(&vec![Action::Discrete(action)]);
        let r = &result.agents[0];
        self.sailed_by = Some(by);
        self.last_action = label.clone();
        self.last_reward = Some(r.reward);
        self.episode_return += r.reward;
        self.outcome = Some(if r.terminated {
            Outcome::ReachedShore
        } else if r.truncated {
            Outcome::OutOfTime
        } else {
            Outcome::Sailing
        });
        self.log = Vec::new();
        if by == SailedBy::RandomAgent {
            self.log
                .push(format!("random agent chose {}", label.unwrap_or_default()));
        }
        self.log.extend(result.info.events.iter().map(|e| match e {
            StepEvent::Moved { to, .. } => format!("moved to ({}, {})", to.0, to.1),
            StepEvent::BumpedWall { .. } => "bumped into the edge · the move is wasted".into(),
            StepEvent::GoalReached { .. } => "reached the shore".into(),
            StepEvent::Reward { term, value, .. } => format!("reward {value:+} ({term})"),
        }));
        match self.outcome {
            Some(Outcome::ReachedShore) => self
                .log
                .push(format!("episode over · return {}", self.episode_return)),
            Some(Outcome::OutOfTime) => self
                .log
                .push("out of time · the episode is truncated".into()),
            _ => {}
        }
    }

    fn snapshot(&self) -> Snapshot {
        let world = self.world.as_ref().expect("snapshot requires a world");
        let spec = &world.spec().agents[0];
        let g = world.grid();
        Snapshot {
            level_id: self.level_id.clone(),
            title: g.title.clone(),
            render: world.render_model(),
            pilot: g.pilot,
            objective: self.objective.clone(),
            budget_episodes: g.budget.as_ref().map(|b| b.episodes),
            random_baseline: self.random_baseline,
            sailed_by: self.sailed_by.unwrap_or(SailedBy::Nobody),
            seed: self.seed,
            episode: self.episode,
            step: world.step_count(),
            max_steps: g.max_steps,
            position: world.position(0),
            state_index: world.observe(0).state_index,
            state_count: spec.observation.state_count(),
            last_action: self.last_action.clone(),
            last_reward: self.last_reward,
            episode_return: self.episode_return,
            outcome: self.outcome.unwrap_or(Outcome::Sailing),
            log: self.log.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(r: SimResponse) -> Snapshot {
        match r {
            SimResponse::Snapshot(s) => *s,
            SimResponse::Error(e) => panic!("{e}"),
        }
    }

    fn load(sim: &mut SimCore, level: &str) -> Snapshot {
        snap(sim.handle(SimRequest::Load {
            level_id: level.into(),
            seed: 7,
        }))
    }

    #[test]
    fn sail_level_zero_to_the_shore() {
        let mut sim = SimCore::default();
        let s = load(&mut sim, "l00_calypsos_loom");
        assert_eq!(
            (s.episode, s.step, s.position, s.state_count),
            (1, 0, (0, 0), Some(25))
        );
        assert_eq!(s.pilot, Pilot::Player);
        assert_eq!(
            s.objective.as_ref().map(|o| o.goal),
            Some(ObjectiveGoal::BestReturn {
                target: 2.0,
                moves: 8
            })
        );
        assert_eq!(s.random_baseline, None);
        let mut last = s;
        for action in [1, 1, 1, 1, 2, 2, 2, 2] {
            last = snap(sim.handle(SimRequest::Step { action }));
        }
        assert_eq!(last.outcome, Outcome::ReachedShore);
        assert_eq!(last.episode_return, 2.0);
        assert_eq!(last.sailed_by, SailedBy::Player);
        assert_eq!(last.last_action.as_deref(), Some("south"));
        let after = snap(sim.handle(SimRequest::Step { action: 0 }));
        assert_eq!(after.step, 8);
        let reset = snap(sim.handle(SimRequest::Reset { seed: 7 }));
        assert_eq!(
            (
                reset.episode,
                reset.step,
                reset.episode_return,
                reset.sailed_by
            ),
            (2, 0, 0.0, SailedBy::Nobody)
        );
    }

    #[test]
    fn level_one_is_agent_piloted() {
        let mut sim = SimCore::default();
        let s = load(&mut sim, "l01_the_raft");
        assert_eq!(s.pilot, Pilot::Agent);
        let baseline = s
            .random_baseline
            .expect("agent levels measure the random agent");
        assert_eq!(baseline.episodes, BASELINE_EPISODES);
        assert!(
            baseline.mean_return < 2.0,
            "random play must fall short of the best return: {baseline:?}"
        );

        // The player cannot steer.
        let steered = snap(sim.handle(SimRequest::Step { action: 1 }));
        assert_eq!((steered.step, steered.position), (0, (0, 0)));
        assert!(steered.log[0].contains("steering is off"));

        // The random agent can, and its episodes are reproducible from the seed.
        let run = |sim: &mut SimCore| {
            sim.handle(SimRequest::Reset { seed: 3 });
            let mut s = snap(sim.handle(SimRequest::AgentStep));
            while s.outcome == Outcome::Sailing {
                s = snap(sim.handle(SimRequest::AgentStep));
            }
            (s.step, s.episode_return, s.sailed_by)
        };
        let a = run(&mut sim);
        assert_eq!(a, run(&mut sim));
        assert_eq!(a.2, SailedBy::RandomAgent);
    }

    #[test]
    fn errors_are_reported_not_panicked() {
        let mut sim = SimCore::default();
        assert!(matches!(
            sim.handle(SimRequest::Step { action: 0 }),
            SimResponse::Error(_)
        ));
        assert!(matches!(
            sim.handle(SimRequest::AgentStep),
            SimResponse::Error(_)
        ));
        assert!(matches!(
            sim.handle(SimRequest::Load {
                level_id: "nope".into(),
                seed: 0
            }),
            SimResponse::Error(_)
        ));
    }
}
