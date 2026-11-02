//! Compilation of a validated `LevelDef` into flat arrays. The hot loop
//! never touches strings or maps.

use rl_core::{ActionSpace, AgentSpec, FeatureSpec, Glyph, ObservationSpec, WorldSpec};

use crate::catalogue::RewardTerm;
use crate::level::{
    BudgetDef, ChallengeDef, Encoding, EngineDef, FeatureDef, LevelDef, Pilot, TerminationDef,
};
use crate::validate::{CellMeaning, LevelErrors, resolve_char, validate};

#[derive(Clone, Debug)]
pub struct CompiledGrid {
    pub id: String,
    pub title: String,
    pub width: i32,
    pub height: i32,
    /// Row-major, `width * height`.
    pub walls: Vec<bool>,
    /// Row-major goal reward term per cell, if the cell is a goal.
    pub goals: Vec<Option<RewardTerm>>,
    pub spawns: Vec<(i32, i32)>,
    pub agent_names: Vec<String>,
    pub agent_glyphs: Vec<Glyph>,
    /// Weight per reward term, indexed by `RewardTerm::index`.
    pub reward_weights: [f32; RewardTerm::COUNT],
    pub terminate_on_goal: bool,
    pub max_steps: u32,
    pub features: Vec<FeatureDef>,
    pub spec: WorldSpec,
    pub pilot: Pilot,
    pub challenge: Option<ChallengeDef>,
    pub budget: Option<BudgetDef>,
}

impl CompiledGrid {
    pub fn compile(def: &LevelDef) -> Result<Self, LevelErrors> {
        validate(def)?;
        let compiled = Self::build(def);
        if matches!(
            compiled.challenge,
            Some(ChallengeDef {
                goal: crate::level::GoalDef::BestReturn,
                ..
            })
        ) && compiled.best_return(0).is_none()
        {
            return Err(LevelErrors(vec![crate::validate::LevelError::new(
                "challenge",
                "BestReturn needs a goal reachable within max_steps",
            )]));
        }
        Ok(compiled)
    }

    fn build(def: &LevelDef) -> Self {
        let EngineDef::Grid { width, height, .. } = def.engine;
        let (w, h) = (width as i32, height as i32);
        let cells = (width * height) as usize;

        let mut walls = vec![false; cells];
        let mut goals = vec![None; cells];
        let mut spawns = vec![(0, 0); def.agents.len()];
        for (y, row) in def.map.iter().enumerate() {
            for (x, c) in row.chars().enumerate() {
                let i = y * width as usize + x;
                match resolve_char(def, c).expect("validated") {
                    CellMeaning::Wall => walls[i] = true,
                    CellMeaning::Goal(term) => goals[i] = RewardTerm::from_name(term),
                    CellMeaning::Spawn(agent) => {
                        let id = def.agents.get_index_of(agent).expect("validated");
                        spawns[id] = (x as i32, y as i32);
                    }
                    CellMeaning::Open => {}
                }
            }
        }

        let mut reward_weights = [0.0; RewardTerm::COUNT];
        for (name, weight) in &def.reward {
            reward_weights[RewardTerm::from_name(name).expect("validated").index()] = *weight;
        }

        let observation = ObservationSpec {
            features: def
                .observation
                .iter()
                .map(|f| feature_spec(f, width, height))
                .collect(),
        };
        let spec = WorldSpec {
            agents: def
                .agents
                .keys()
                .map(|name| AgentSpec {
                    name: name.clone(),
                    observation: observation.clone(),
                    action: ActionSpace::Discrete {
                        n: 4,
                        labels: ["north", "east", "south", "west"]
                            .map(String::from)
                            .to_vec(),
                    },
                })
                .collect(),
        };

        Self {
            id: def.id.clone(),
            title: def.title.clone(),
            width: w,
            height: h,
            walls,
            goals,
            spawns,
            agent_names: def.agents.keys().cloned().collect(),
            agent_glyphs: def.agents.values().map(|a| a.glyph).collect(),
            reward_weights,
            terminate_on_goal: def.termination.contains(&TerminationDef::GoalReached),
            max_steps: def.truncation.max_steps,
            features: def.observation.clone(),
            spec,
            pilot: def.pilot,
            challenge: def.challenge.clone(),
            budget: def.budget.clone(),
        }
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

    /// Fewest moves from `agent`'s spawn to any goal cell, by breadth-first
    /// search, together with that goal's reward term. `None` if no goal is
    /// reachable.
    pub fn shortest_path_to_goal(&self, agent: usize) -> Option<(u32, RewardTerm)> {
        let start = self.spawns[agent];
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
                if self.in_bounds(next) && !self.is_wall(next) && dist[self.index(next)] == u32::MAX
                {
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
        (moves <= self.max_steps).then(|| {
            (
                moves as f32 * self.weight(RewardTerm::Step) + self.weight(term),
                moves,
            )
        })
    }
}

fn feature_spec(f: &FeatureDef, width: u32, height: u32) -> FeatureSpec {
    let cells = width * height;
    match f {
        FeatureDef::SelfPosition { encoding } => match encoding {
            Encoding::Index => FeatureSpec {
                name: "self_position(encoding: index)".into(),
                size: 1,
                cardinality: Some(cells),
            },
            Encoding::Normalised => FeatureSpec {
                name: "self_position(encoding: normalised)".into(),
                size: 2,
                cardinality: None,
            },
            Encoding::OneHot => FeatureSpec {
                name: "self_position(encoding: one_hot)".into(),
                size: cells,
                cardinality: Some(cells),
            },
        },
    }
}
