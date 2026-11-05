//! Runtime grid world.

use rl_core::{
    Action, ActionMask, AgentId, JointAction, Observation, PolicyProvider, RenderAgent, RenderEvent, RenderModel, Rng,
    StepEvent, StepInfo, StepResult, StepResults, World, WorldSpec,
};

use super::compiled::{CompiledFeature, CompiledGrid};
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
    pub const ALL: [Direction; 4] = [Direction::North, Direction::East, Direction::South, Direction::West];

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
    /// The action taken last step, for the `previous_action` feature.
    last_action: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct GridWorld {
    grid: CompiledGrid,
    agents: Vec<AgentState>,
    steps: u32,
    /// Seeded on reset. Picks spawn cells; later also wind and other noise.
    rng: Rng,
    last_events: Vec<StepEvent>,
}

impl GridWorld {
    pub fn compile(def: &LevelDef) -> Result<Self, LevelErrors> {
        let grid = CompiledGrid::compile(def)?;
        let mut world = Self { agents: Vec::new(), steps: 0, rng: Rng::new(0), last_events: Vec::new(), grid };
        world.reset(0);
        Ok(world)
    }

    pub fn grid(&self) -> &CompiledGrid {
        &self.grid
    }

    /// Changes one reward weight for this world (the player's reward editor).
    pub fn set_reward_weight(&mut self, term: RewardTerm, weight: f32) {
        self.grid.set_weight(term, weight);
    }

    /// Replaces the agent's observation features. Callers check the level's
    /// editing rules; see [`GridWorld::apply_edits`].
    pub fn set_observation(&mut self, features: Vec<FeatureDef>) {
        self.grid.set_features(features);
    }

    pub fn position(&self, agent: AgentId) -> (i32, i32) {
        self.agents[agent].position
    }

    /// Whether every learner's episode is over. Scripted agents never end it.
    pub fn is_done(&self) -> bool {
        self.agents[..self.grid.learners].iter().all(|a| a.terminated || a.truncated)
    }

    /// Positions of every agent, scripted ones included, in file order.
    pub fn positions(&self) -> Vec<(i32, i32)> {
        self.agents.iter().map(|a| a.position).collect()
    }

    /// The current observation of learner `agent`, built from the level's observation spec.
    pub fn observe(&self, agent: AgentId) -> Observation {
        self.observe_at(agent, self.agents[agent].position)
    }

    /// What learner `agent` would observe at `cell`, everything else as it is
    /// now. The arena uses it to draw the policy for the current situation.
    pub fn observe_at(&self, agent: AgentId, cell: (i32, i32)) -> Observation {
        let g = &self.grid;
        let (x, y) = cell;
        let cells = (g.width * g.height) as u32;
        let cell_of = |(x, y): (i32, i32)| (y * g.width + x) as u32;
        let mut features = Vec::new();
        let mut state_index: Option<u32> = Some(0);
        // A discrete feature adds one number and multiplies the state space by its cardinality.
        let discrete = |features: &mut Vec<f32>, state: Option<u32>, value: u32, cardinality: u32| {
            features.push(value as f32);
            state.map(|s| s * cardinality + value)
        };
        for f in &g.compiled_features {
            match *f {
                CompiledFeature::SelfPosition(encoding) => {
                    let cell = cell_of((x, y));
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
                CompiledFeature::AgentPosition(other) => {
                    state_index = discrete(&mut features, state_index, cell_of(self.agents[other].position), cells);
                }
                CompiledFeature::AgentDirection(other) => {
                    let (ox, oy) = self.agents[other].position;
                    let sector = direction_sector(ox - x, oy - y);
                    state_index = discrete(&mut features, state_index, sector, super::compiled::DIRECTION_VALUES);
                }
                CompiledFeature::PreviousAction => {
                    let value = self.agents[agent].last_action.map_or(0, |a| a + 1);
                    state_index = discrete(&mut features, state_index, value, super::compiled::PREVIOUS_ACTION_VALUES);
                }
            }
        }
        Observation { features, state_index }
    }

    fn observe_all(&self) -> Vec<Observation> {
        (0..self.grid.learners).map(|a| self.observe(a)).collect()
    }

    /// Learners not yet terminated or truncated, sharing a cell with a scripted agent.
    fn catch(&mut self, events: &mut Vec<StepEvent>, rewards: &mut [f32]) {
        let catchers: Vec<(usize, (i32, i32))> =
            self.grid.scripted.iter().map(|s| (s.agent, self.agents[s.agent].position)).collect();
        let value = self.grid.weight(RewardTerm::CaughtByOpponent);
        let learners = self.agents[..self.grid.learners].iter_mut().zip(rewards.iter_mut()).enumerate();
        for (id, (a, reward)) in learners {
            if a.terminated || a.truncated {
                continue;
            }
            let Some(&(by, at)) = catchers.iter().find(|(_, p)| *p == a.position) else { continue };
            events.push(StepEvent::Caught { agent: id, by, at });
            if value != 0.0 {
                *reward += value;
                events.push(StepEvent::Reward { agent: id, term: RewardTerm::CaughtByOpponent.name().into(), value });
            }
            if self.grid.terminate_on_catch {
                a.terminated = true;
            }
        }
    }

    /// Scripted agents take their move, each through its behaviour's policy.
    fn move_scripted(&mut self, events: &mut Vec<StepEvent>) {
        for k in 0..self.grid.scripted.len() {
            let s = &self.grid.scripted[k];
            if !self.steps.is_multiple_of(s.every_n_steps.max(1)) {
                continue;
            }
            let me = s.agent;
            let from = self.agents[me].position;
            // It chases the first learner still sailing: every level so far has one.
            let target = (0..self.grid.learners)
                .find(|&l| !self.agents[l].terminated && !self.agents[l].truncated)
                .map(|l| self.agents[l].position);
            let Some(target) = target else { continue };
            let (from_cell, target_cell) = (self.grid.index(from), self.grid.index(target));
            if s.chase.first_move(from_cell, target_cell).is_none() {
                continue;
            }
            // The behaviour is a policy: it sees where it is and where its quarry is, and picks a move.
            let mut policy: &crate::behaviours::ChaseShortestPath = &s.chase;
            let observation = policy.observation(from_cell, target_cell);
            let action = policy.act(&observation, &ActionMask::all(4), &mut Rng::new(0));
            let dir = Direction::from_action(action);
            let (dx, dy) = dir.delta();
            let to = (from.0 + dx, from.1 + dy);
            if self.grid.in_bounds(to) && !self.grid.is_wall(to) {
                self.agents[me].position = to;
                self.agents[me].facing = dir;
                events.push(StepEvent::Moved { agent: me, from, to });
            }
        }
    }
}

/// Which way an offset points: 0 here, then 1 north, 2 east, 3 south, 4 west,
/// by the larger of the two offsets (vertical on a tie).
fn direction_sector(dx: i32, dy: i32) -> u32 {
    match (dx, dy) {
        (0, 0) => 0,
        _ if dy.abs() >= dx.abs() => {
            if dy < 0 {
                1
            } else {
                3
            }
        }
        _ if dx > 0 => 2,
        _ => 4,
    }
}

impl World for GridWorld {
    fn reset(&mut self, seed: u64) -> Vec<Observation> {
        self.rng = Rng::new(seed);
        self.steps = 0;
        self.last_events.clear();
        let spawns = self.grid.spawns.clone();
        self.agents = spawns
            .iter()
            .map(|options| {
                // A single spawn cell draws nothing, so levels without choices replay as before.
                let position = if options.len() == 1 {
                    options[0]
                } else {
                    options[self.rng.below(options.len() as u32) as usize]
                };
                AgentState { position, facing: Direction::East, terminated: false, truncated: false, last_action: None }
            })
            .collect();
        self.observe_all()
    }

    fn step(&mut self, actions: &JointAction) -> StepResults {
        let learners = self.grid.learners;
        assert_eq!(actions.len(), learners, "one action per learner is required");
        let mut events = Vec::new();
        let mut rewards = vec![0.0f32; learners];

        if !self.is_done() {
            self.steps += 1;
            for (id, action) in actions.iter().enumerate() {
                if self.agents[id].terminated || self.agents[id].truncated {
                    continue;
                }
                let dir = Direction::from_action(*action);
                self.agents[id].last_action = Some(dir as u32);
                let from = self.agents[id].position;
                let (dx, dy) = dir.delta();
                let to = (from.0 + dx, from.1 + dy);
                self.agents[id].facing = dir;
                if self.grid.in_bounds(to) && !self.grid.is_wall(to) {
                    self.agents[id].position = to;
                    events.push(StepEvent::Moved { agent: id, from, to });
                } else {
                    events.push(StepEvent::BumpedWall { agent: id, at: to });
                }

                let mut pay = |term: RewardTerm, events: &mut Vec<StepEvent>| {
                    let value = self.grid.weight(term);
                    if value != 0.0 {
                        rewards[id] += value;
                        events.push(StepEvent::Reward { agent: id, term: term.name().into(), value });
                    }
                };
                pay(RewardTerm::Step, &mut events);

                let here = self.agents[id].position;
                let cell = self.grid.index(here);
                if self.grid.lethal[cell] {
                    events.push(StepEvent::Died { agent: id, at: here });
                    pay(RewardTerm::Death, &mut events);
                    self.agents[id].terminated = true;
                    continue;
                }
                if self.grid.hazards[cell] {
                    events.push(StepEvent::OnHazard { agent: id, at: here });
                    pay(RewardTerm::Hazard, &mut events);
                }
                if self.grid.lures[cell] {
                    events.push(StepEvent::OnLure { agent: id, at: here });
                    pay(RewardTerm::Lure, &mut events);
                }
                if let Some(term) = self.grid.goals[self.grid.index(here)] {
                    events.push(StepEvent::GoalReached { agent: id, at: here });
                    pay(term, &mut events);
                    if self.grid.terminate_on_goal {
                        self.agents[id].terminated = true;
                    }
                }
            }
            // A learner that walks into an opponent is caught; then the
            // opponents move, and may catch a learner where it stands.
            if !self.grid.scripted.is_empty() {
                self.catch(&mut events, &mut rewards);
                self.move_scripted(&mut events);
                self.catch(&mut events, &mut rewards);
            }
            if self.steps >= self.grid.max_steps {
                for a in &mut self.agents[..learners] {
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
                .zip(&self.agents[..learners])
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
            hazards: cells(&|i| g.hazards[i]),
            lethal: cells(&|i| g.lethal[i]),
            lures: cells(&|i| g.lures[i]),
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
                        Some(RenderEvent::Reward { position: centre(self.agents[*agent].position), value: *value })
                    }
                    StepEvent::BumpedWall { at, .. } => Some(RenderEvent::Bump { position: centre(*at) }),
                    _ => None,
                })
                .collect(),
        }
    }
}
