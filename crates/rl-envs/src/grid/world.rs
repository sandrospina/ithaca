//! Runtime grid world.

use rl_core::{
    Action, ActionMask, AgentId, JointAction, Observation, RenderAgent, RenderEvent, RenderModel,
    Rng, StepEvent, StepInfo, StepResult, StepResults, World, WorldSpec,
};

use super::compiled::CompiledGrid;
use crate::catalogue::RewardTerm;
use crate::level::{Encoding, FeatureDef, LevelDef};
use crate::validate::LevelErrors;

/// Discrete grid actions. The action index is the enum discriminant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
}

impl Direction {
    pub const ALL: [Direction; 4] = [
        Direction::North,
        Direction::East,
        Direction::South,
        Direction::West,
    ];

    pub fn from_action(action: Action) -> Self {
        Self::ALL[(action.discrete() % 4) as usize]
    }

    pub fn action(self) -> Action {
        Action::Discrete(self as u32)
    }

    pub fn delta(self) -> (i32, i32) {
        match self {
            Direction::North => (0, -1),
            Direction::East => (1, 0),
            Direction::South => (0, 1),
            Direction::West => (-1, 0),
        }
    }

    /// Screen heading in radians: 0 = east, clockwise positive.
    pub fn heading(self) -> f32 {
        use std::f32::consts::{FRAC_PI_2, PI};
        match self {
            Direction::North => -FRAC_PI_2,
            Direction::East => 0.0,
            Direction::South => FRAC_PI_2,
            Direction::West => PI,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AgentState {
    position: (i32, i32),
    facing: Direction,
    terminated: bool,
    truncated: bool,
}

#[derive(Clone, Debug)]
pub struct GridWorld {
    grid: CompiledGrid,
    agents: Vec<AgentState>,
    steps: u32,
    /// Reserved for stochastic mechanics (wind, random spawns) from Level 4.
    rng: Rng,
    last_events: Vec<StepEvent>,
}

impl GridWorld {
    pub fn compile(def: &LevelDef) -> Result<Self, LevelErrors> {
        let grid = CompiledGrid::compile(def)?;
        let mut world = Self {
            agents: Vec::new(),
            steps: 0,
            rng: Rng::new(0),
            last_events: Vec::new(),
            grid,
        };
        world.reset(0);
        Ok(world)
    }

    pub fn grid(&self) -> &CompiledGrid {
        &self.grid
    }

    pub fn position(&self, agent: AgentId) -> (i32, i32) {
        self.agents[agent].position
    }

    pub fn is_done(&self) -> bool {
        self.agents.iter().all(|a| a.terminated || a.truncated)
    }

    /// The current observation of `agent`, built from the level's observation spec.
    pub fn observe(&self, agent: AgentId) -> Observation {
        let g = &self.grid;
        let (x, y) = self.agents[agent].position;
        let cells = (g.width * g.height) as u32;
        let mut features = Vec::new();
        let mut state_index: Option<u32> = Some(0);
        for f in &g.features {
            match f {
                FeatureDef::SelfPosition { encoding } => {
                    let cell = (y * g.width + x) as u32;
                    match encoding {
                        Encoding::Index => features.push(cell as f32),
                        Encoding::Normalised => {
                            features.push(x as f32 / (g.width - 1).max(1) as f32);
                            features.push(y as f32 / (g.height - 1).max(1) as f32);
                        }
                        Encoding::OneHot => {
                            let start = features.len();
                            features.resize(start + cells as usize, 0.0);
                            features[start + cell as usize] = 1.0;
                        }
                    }
                    state_index = match encoding {
                        Encoding::Index | Encoding::OneHot => state_index.map(|s| s * cells + cell),
                        Encoding::Normalised => None,
                    };
                }
            }
        }
        Observation {
            features,
            state_index,
        }
    }

    fn observe_all(&self) -> Vec<Observation> {
        (0..self.agents.len()).map(|a| self.observe(a)).collect()
    }
}

impl World for GridWorld {
    fn reset(&mut self, seed: u64) -> Vec<Observation> {
        self.rng = Rng::new(seed);
        self.steps = 0;
        self.last_events.clear();
        self.agents = self
            .grid
            .spawns
            .iter()
            .map(|&position| AgentState {
                position,
                facing: Direction::East,
                terminated: false,
                truncated: false,
            })
            .collect();
        self.observe_all()
    }

    fn step(&mut self, actions: &JointAction) -> StepResults {
        assert_eq!(
            actions.len(),
            self.agents.len(),
            "one action per agent is required"
        );
        let mut events = Vec::new();
        let mut rewards = vec![0.0f32; self.agents.len()];

        if !self.is_done() {
            self.steps += 1;
            for (id, action) in actions.iter().enumerate() {
                if self.agents[id].terminated || self.agents[id].truncated {
                    continue;
                }
                let dir = Direction::from_action(*action);
                let from = self.agents[id].position;
                let (dx, dy) = dir.delta();
                let to = (from.0 + dx, from.1 + dy);
                self.agents[id].facing = dir;
                if self.grid.in_bounds(to) && !self.grid.is_wall(to) {
                    self.agents[id].position = to;
                    events.push(StepEvent::Moved {
                        agent: id,
                        from,
                        to,
                    });
                } else {
                    events.push(StepEvent::BumpedWall { agent: id, at: to });
                }

                let mut pay = |term: RewardTerm, events: &mut Vec<StepEvent>| {
                    let value = self.grid.weight(term);
                    if value != 0.0 {
                        rewards[id] += value;
                        events.push(StepEvent::Reward {
                            agent: id,
                            term: term.name().into(),
                            value,
                        });
                    }
                };
                pay(RewardTerm::Step, &mut events);

                let here = self.agents[id].position;
                if let Some(term) = self.grid.goals[self.grid.index(here)] {
                    events.push(StepEvent::GoalReached {
                        agent: id,
                        at: here,
                    });
                    pay(term, &mut events);
                    if self.grid.terminate_on_goal {
                        self.agents[id].terminated = true;
                    }
                }
            }
            if self.steps >= self.grid.max_steps {
                for a in &mut self.agents {
                    if !a.terminated {
                        a.truncated = true;
                    }
                }
            }
        }

        self.last_events = events.clone();
        let observations = self.observe_all();
        StepResults {
            agents: observations
                .into_iter()
                .zip(&self.agents)
                .zip(rewards)
                .map(|((observation, a), reward)| StepResult {
                    observation,
                    reward,
                    terminated: a.terminated,
                    truncated: a.truncated,
                })
                .collect(),
            info: StepInfo { events },
        }
    }

    fn action_mask(&self, _agent: AgentId) -> ActionMask {
        // Bumping into walls is allowed and teaches the agent; every move is valid.
        ActionMask::all(4)
    }

    fn spec(&self) -> &WorldSpec {
        &self.grid.spec
    }

    fn step_count(&self) -> u32 {
        self.steps
    }

    fn render_model(&self) -> RenderModel {
        let g = &self.grid;
        let cells = |pred: &dyn Fn(usize) -> bool| -> Vec<(i32, i32)> {
            (0..g.height)
                .flat_map(|y| (0..g.width).map(move |x| (x, y)))
                .filter(|&(x, y)| pred((y * g.width + x) as usize))
                .collect()
        };
        let centre = |(x, y): (i32, i32)| (x as f32 + 0.5, y as f32 + 0.5);
        RenderModel {
            width: g.width as f32,
            height: g.height as f32,
            walls: cells(&|i| g.walls[i]),
            goals: cells(&|i| g.goals[i].is_some()),
            agents: self
                .agents
                .iter()
                .enumerate()
                .map(|(i, a)| RenderAgent {
                    name: g.agent_names[i].clone(),
                    position: centre(a.position),
                    heading: a.facing.heading(),
                    glyph: g.agent_glyphs[i],
                })
                .collect(),
            events: self
                .last_events
                .iter()
                .filter_map(|e| match e {
                    StepEvent::Reward { agent, term, value } if term != RewardTerm::Step.name() => {
                        Some(RenderEvent::Reward {
                            position: centre(self.agents[*agent].position),
                            value: *value,
                        })
                    }
                    StepEvent::BumpedWall { at, .. } => Some(RenderEvent::Bump {
                        position: centre(*at),
                    }),
                    _ => None,
                })
                .collect(),
        }
    }
}
