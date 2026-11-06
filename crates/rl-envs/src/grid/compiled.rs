//! Compilation of a validated `LevelDef` into flat arrays. The hot loop
//! never touches strings or maps.

use std::sync::Arc;

use rl_core::{ActionSpace, AgentSpec, FeatureSpec, Glyph, ObservationSpec, WorldSpec};

use crate::behaviours::{BehaviourDef, ChaseShortestPath};
use crate::catalogue::RewardTerm;
use crate::level::{
    BudgetDef, ChallengeDef, ControllerDef, Encoding, EngineDef, FeatureDef, LevelDef, Pilot, TerminationDef,
};
use crate::validate::{CellMeaning, LevelErrors, resolve_char, validate};

#[derive(Clone, Debug)]
pub struct CompiledGrid {
    pub id: String,
    pub title: String,
    /// What the goal is called in messages.
    pub goal_name: String,
    pub width: i32,
    pub height: i32,
    /// Row-major, `width * height`.
    pub walls: Vec<bool>,
    /// Row-major goal reward term per cell, if the cell is a goal.
    pub goals: Vec<Option<RewardTerm>>,
    /// Row-major: cells that pay the hazard term each step.
    pub hazards: Vec<bool>,
    /// Row-major: cells that end the episode.
    pub lethal: Vec<bool>,
    /// Row-major: cells that pay the lure term each step.
    pub lures: Vec<bool>,
    /// Row-major: windy cells, where a move may be turned aside.
    pub winds: Vec<bool>,
    /// Chance a move from a windy cell is turned aside, now.
    pub slip: f32,
    /// The level's own slip: what tests always sail in.
    pub level_slip: f32,
    /// Editor label per reward term, from the level or the catalogue.
    pub reward_labels: [String; RewardTerm::COUNT],
    /// Reward terms present in the level file, in file order.
    pub reward_terms: Vec<RewardTerm>,
    pub player_editable: crate::level::PlayerEditable,
    /// Spawn cells per agent. With several, each episode picks one from its seed.
    pub spawns: Vec<Vec<(i32, i32)>>,
    /// Every agent, learners first.
    pub agent_names: Vec<String>,
    /// How many agents are learners, controlled from outside. They come first.
    pub learners: usize,
    /// Agents driven by a coded behaviour, after the learners.
    pub scripted: Vec<ScriptedAgent>,
    pub agent_glyphs: Vec<Glyph>,
    /// Weight per reward term, indexed by `RewardTerm::index`.
    pub reward_weights: [f32; RewardTerm::COUNT],
    pub terminate_on_goal: bool,
    pub terminate_on_catch: bool,
    pub max_steps: u32,
    pub features: Vec<FeatureDef>,
    /// `features` with agent names resolved to indices, for the hot loop.
    pub compiled_features: Vec<CompiledFeature>,
    pub spec: WorldSpec,
    pub pilot: Pilot,
    pub challenge: Option<ChallengeDef>,
    pub budget: Option<BudgetDef>,
}

impl CompiledGrid {
    pub fn compile(def: &LevelDef) -> Result<Self, LevelErrors> {
        validate(def)?;
        let compiled = Self::build(def);
        let has_specials = compiled.hazards.iter().chain(&compiled.lures).chain(&compiled.lethal).any(|&b| b);
        if compiled.needs_best_return() && has_specials {
            return Err(LevelErrors(vec![crate::validate::LevelError::new(
                "challenge",
                "shortest-route measures are only defined for maps without hazards or lures",
            )]));
        }
        if compiled.needs_best_return() && compiled.best_return(0).is_none() {
            return Err(LevelErrors(vec![crate::validate::LevelError::new(
                "challenge",
                "this challenge measures the shortest route, so a goal must be reachable within max_steps",
            )]));
        }
        Ok(compiled)
    }

    /// Whether the challenge refers to the best possible return.
    fn needs_best_return(&self) -> bool {
        use crate::level::{BatchStat, GoalDef, QuestionDef, SuccessDef, TaskDef};
        match self.challenge.as_ref().map(|c| &c.goal) {
            Some(GoalDef::BestReturn) => true,
            Some(GoalDef::Tasks(tasks)) => tasks.iter().any(|t| match t {
                TaskDef::TrainAgent { success: SuccessDef::ShortestRoute, .. } => true,
                TaskDef::Quiz { questions } => {
                    questions.iter().any(|q| matches!(q, QuestionDef::StatBand { stat: BatchStat::ShortestRoute, .. }))
                }
                _ => false,
            }),
            _ => false,
        }
    }

    fn build(def: &LevelDef) -> Self {
        let EngineDef::Grid { width, height, .. } = def.engine;
        let (w, h) = (width as i32, height as i32);
        let cells = (width * height) as usize;

        let mut walls = vec![false; cells];
        let mut goals = vec![None; cells];
        let mut hazards = vec![false; cells];
        let mut lethal = vec![false; cells];
        let mut lures = vec![false; cells];
        let mut winds = vec![false; cells];
        let mut spawns = vec![Vec::new(); def.agents.len()];
        for (y, row) in def.map.iter().enumerate() {
            for (x, c) in row.chars().enumerate() {
                let i = y * width as usize + x;
                match resolve_char(def, c).expect("validated") {
                    CellMeaning::Wall => walls[i] = true,
                    CellMeaning::Goal(term) => goals[i] = RewardTerm::from_name(term),
                    CellMeaning::Hazard { lethal: true } => lethal[i] = true,
                    CellMeaning::Hazard { lethal: false } => hazards[i] = true,
                    CellMeaning::Lure => lures[i] = true,
                    CellMeaning::Wind => winds[i] = true,
                    CellMeaning::Spawn(agent) => {
                        let id = def.agents.get_index_of(agent).expect("validated");
                        spawns[id].push((x as i32, y as i32));
                    }
                    CellMeaning::Open => {}
                }
            }
        }

        let mut reward_weights = [0.0; RewardTerm::COUNT];
        for (name, weight) in &def.reward {
            reward_weights[RewardTerm::from_name(name).expect("validated").index()] = *weight;
        }

        let agent_names: Vec<String> = def.agents.keys().cloned().collect();
        let learners = def.agents.values().take_while(|a| a.controller == ControllerDef::Learner).count();
        let spec = world_spec(&agent_names[..learners], &def.observation, width, height);
        let chase = Arc::new(ChaseShortestPath::new(w, h, &walls));
        let scripted = def
            .agents
            .values()
            .enumerate()
            .filter_map(|(agent, a)| match &a.controller {
                ControllerDef::Learner => None,
                ControllerDef::Scripted { behaviour: BehaviourDef::ChaseShortestPath { every_n_steps, p_random } } => {
                    Some(ScriptedAgent {
                        agent,
                        every_n_steps: *every_n_steps,
                        p_random: *p_random,
                        chase: chase.clone(),
                    })
                }
            })
            .collect();

        Self {
            id: def.id.clone(),
            title: def.title.clone(),
            goal_name: def.goal_name.clone().unwrap_or_else(|| "the shore".into()),
            width: w,
            height: h,
            walls,
            goals,
            hazards,
            lethal,
            lures,
            winds,
            slip: def.wind.map_or(0.0, |w| w.slip),
            level_slip: def.wind.map_or(0.0, |w| w.slip),
            reward_labels: RewardTerm::ALL
                .map(|t| def.reward_labels.get(t.name()).cloned().unwrap_or_else(|| t.description().to_string())),
            reward_terms: def.reward.keys().filter_map(|n| RewardTerm::from_name(n)).collect(),
            player_editable: def.player_editable.clone(),
            spawns,
            compiled_features: compile_features(&def.observation, &agent_names),
            agent_names,
            learners,
            scripted,
            agent_glyphs: def.agents.values().map(|a| a.glyph).collect(),
            reward_weights,
            terminate_on_goal: def.termination.contains(&TerminationDef::GoalReached),
            terminate_on_catch: def.termination.contains(&TerminationDef::CaughtByOpponent),
            max_steps: def.truncation.max_steps,
            features: def.observation.clone(),
            spec,
            pilot: def.pilot,
            challenge: def.challenge.clone(),
            budget: def.budget.clone(),
        }
    }

    /// Replaces every learner's observation features, e.g. from the
    /// Observation Builder. Callers check the level's editing rules.
    pub fn set_features(&mut self, features: Vec<FeatureDef>) {
        self.spec = world_spec(&self.agent_names[..self.learners], &features, self.width as u32, self.height as u32);
        self.compiled_features = compile_features(&features, &self.agent_names);
        self.features = features;
    }

    /// Distinct values of a feature in this grid, if it is discrete.
    pub fn feature_cardinality(&self, f: &FeatureDef) -> Option<u32> {
        feature_spec(f, self.width as u32, self.height as u32).cardinality
    }

    pub fn index(&self, (x, y): (i32, i32)) -> usize {
        (y * self.width + x) as usize
    }

    pub fn in_bounds(&self, (x, y): (i32, i32)) -> bool {
        (0..self.width).contains(&x) && (0..self.height).contains(&y)
    }

    pub fn is_wall(&self, cell: (i32, i32)) -> bool {
        self.walls[self.index(cell)]
    }

    pub fn weight(&self, term: RewardTerm) -> f32 {
        self.reward_weights[term.index()]
    }

    /// Changes one reward weight, e.g. from the player's reward editor.
    /// Callers enforce the level's editing bounds.
    pub fn set_weight(&mut self, term: RewardTerm, weight: f32) {
        self.reward_weights[term.index()] = weight;
    }

    /// Fewest moves from `agent`'s spawn to any goal cell, by breadth-first
    /// search, together with that goal's reward term. `None` if no goal is
    /// reachable.
    pub fn shortest_path_to_goal(&self, agent: usize) -> Option<(u32, RewardTerm)> {
        let start = self.spawns[agent][0];
        let mut dist = vec![u32::MAX; self.walls.len()];
        let mut queue = std::collections::VecDeque::from([start]);
        dist[self.index(start)] = 0;
        while let Some(cell) = queue.pop_front() {
            let d = dist[self.index(cell)];
            if let Some(term) = self.goals[self.index(cell)]
                && d > 0
            {
                return Some((d, term));
            }
            for (dx, dy) in [(0, -1), (1, 0), (0, 1), (-1, 0)] {
                let next = (cell.0 + dx, cell.1 + dy);
                if self.in_bounds(next) && !self.is_wall(next) && dist[self.index(next)] == u32::MAX {
                    dist[self.index(next)] = d + 1;
                    queue.push_back(next);
                }
            }
        }
        None
    }

    /// The best possible return for `agent`: the shortest path to a goal.
    /// Valid when the step reward is zero or negative, which validation
    /// enforces for `BestReturn` challenges.
    pub fn best_return(&self, agent: usize) -> Option<(f32, u32)> {
        let (moves, term) = self.shortest_path_to_goal(agent)?;
        (moves <= self.max_steps).then(|| (moves as f32 * self.weight(RewardTerm::Step) + self.weight(term), moves))
    }
}

/// A scripted agent and its compiled behaviour.
#[derive(Clone, Debug)]
pub struct ScriptedAgent {
    /// Index among all agents.
    pub agent: usize,
    pub every_n_steps: u32,
    /// Chance each move goes in a random direction instead.
    pub p_random: f32,
    /// Shared between copies of the world: it never changes after compiling.
    pub chase: Arc<ChaseShortestPath>,
}

/// An observation feature with agent names resolved to indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompiledFeature {
    SelfPosition(Encoding),
    AgentPosition(usize),
    AgentDirection(usize),
    PreviousAction,
}

/// Values of the `agent_direction` feature: here, then the four compass sectors.
pub const DIRECTION_VALUES: u32 = 5;
/// Values of the `previous_action` feature: none, then the four moves.
pub const PREVIOUS_ACTION_VALUES: u32 = 5;

fn compile_features(features: &[FeatureDef], agent_names: &[String]) -> Vec<CompiledFeature> {
    let index = |name: &str| agent_names.iter().position(|n| n == name).expect("validated");
    features
        .iter()
        .map(|f| match f {
            FeatureDef::SelfPosition { encoding } => CompiledFeature::SelfPosition(*encoding),
            FeatureDef::AgentPosition { agent } => CompiledFeature::AgentPosition(index(agent)),
            FeatureDef::AgentDirection { agent } => CompiledFeature::AgentDirection(index(agent)),
            FeatureDef::PreviousAction => CompiledFeature::PreviousAction,
        })
        .collect()
}

fn world_spec(agent_names: &[String], features: &[FeatureDef], width: u32, height: u32) -> WorldSpec {
    let observation = ObservationSpec { features: features.iter().map(|f| feature_spec(f, width, height)).collect() };
    WorldSpec {
        agents: agent_names
            .iter()
            .map(|name| AgentSpec {
                name: name.clone(),
                observation: observation.clone(),
                action: ActionSpace::Discrete {
                    n: 4,
                    labels: ["north", "east", "south", "west"].map(String::from).to_vec(),
                },
            })
            .collect(),
    }
}

fn feature_spec(f: &FeatureDef, width: u32, height: u32) -> FeatureSpec {
    let cells = width * height;
    match f {
        FeatureDef::SelfPosition { encoding } => match encoding {
            Encoding::Index => {
                FeatureSpec { name: "self_position(encoding: index)".into(), size: 1, cardinality: Some(cells) }
            }
            Encoding::Normalised => {
                FeatureSpec { name: "self_position(encoding: normalised)".into(), size: 2, cardinality: None }
            }
            Encoding::OneHot => {
                FeatureSpec { name: "self_position(encoding: one_hot)".into(), size: cells, cardinality: Some(cells) }
            }
        },
        FeatureDef::AgentPosition { agent } => {
            FeatureSpec { name: format!("agent_position(agent: {agent})"), size: 1, cardinality: Some(cells) }
        }
        FeatureDef::AgentDirection { agent } => FeatureSpec {
            name: format!("agent_direction(agent: {agent})"),
            size: 1,
            cardinality: Some(DIRECTION_VALUES),
        },
        FeatureDef::PreviousAction => {
            FeatureSpec { name: "previous_action".into(), size: 1, cardinality: Some(PREVIOUS_ACTION_VALUES) }
        }
    }
}
