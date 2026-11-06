//! Scripted behaviours: coded, parameterised opponents selected by name in a
//! level file. Each is a [`PolicyProvider`], so scripted opponents and trained
//! policies plug into a world the same way.

use rl_core::{Action, ActionMask, Observation, PolicyProvider, Rng};
use serde::{Deserialize, Serialize};

use crate::grid::Direction;

/// Behaviours a level file may give a scripted agent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BehaviourDef {
    /// Moves one cell along a shortest path toward the nearest learner,
    /// once every `every_n_steps` steps. With probability `p_random` a move
    /// goes in a random direction instead.
    ChaseShortestPath {
        every_n_steps: u32,
        #[serde(default)]
        p_random: f32,
    },
}

/// `chase_shortest_path`, compiled for one map: for every (from, target)
/// cell pair, the first move of a shortest path, found once at load time.
///
/// Its observation is a single state index, `from * cells + target`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChaseShortestPath {
    cells: u32,
    /// `first_move[from * cells + target]`: a direction, or `None` when the
    /// target is unreachable or already reached.
    first_move: Vec<Option<Direction>>,
}

impl ChaseShortestPath {
    /// Builds the table for a `width × height` grid with the given walls.
    pub fn new(width: i32, height: i32, walls: &[bool]) -> Self {
        let cells = (width * height) as usize;
        let mut first_move = vec![None; cells * cells];
        let index = |(x, y): (i32, i32)| (y * width + x) as usize;
        let open = |(x, y): (i32, i32)| (0..width).contains(&x) && (0..height).contains(&y) && !walls[index((x, y))];
        // A breadth-first search from each target gives every cell's distance
        // to it; the first move is then any step that gets one closer. Ties go
        // to the first direction in north, east, south, west order.
        for target in 0..cells {
            let t = ((target as i32) % width, (target as i32) / width);
            if !open(t) {
                continue;
            }
            let mut dist = vec![u32::MAX; cells];
            dist[target] = 0;
            let mut queue = std::collections::VecDeque::from([t]);
            while let Some(c) = queue.pop_front() {
                for d in Direction::ALL {
                    let (dx, dy) = d.delta();
                    let n = (c.0 + dx, c.1 + dy);
                    if open(n) && dist[index(n)] == u32::MAX {
                        dist[index(n)] = dist[index(c)] + 1;
                        queue.push_back(n);
                    }
                }
            }
            for from in 0..cells {
                let f = ((from as i32) % width, (from as i32) / width);
                if !open(f) || dist[from] == 0 || dist[from] == u32::MAX {
                    continue;
                }
                first_move[from * cells + target] = Direction::ALL.into_iter().find(|d| {
                    let (dx, dy) = d.delta();
                    let n = (f.0 + dx, f.1 + dy);
                    open(n) && dist[index(n)] + 1 == dist[from]
                });
            }
        }
        Self { cells: cells as u32, first_move }
    }

    /// The observation this behaviour reads: where it is and where its target is.
    pub fn observation(&self, from: usize, target: usize) -> Observation {
        Observation { features: Vec::new(), state_index: Some(from as u32 * self.cells + target as u32) }
    }

    /// The move toward `target`, or `None` if there is nowhere to go.
    pub fn first_move(&self, from: usize, target: usize) -> Option<Direction> {
        self.first_move[from * self.cells as usize + target]
    }
}

/// The table never changes, so the policy is a shared reference: copies of a
/// world share one table.
impl PolicyProvider for &ChaseShortestPath {
    fn act(&mut self, observation: &Observation, _mask: &ActionMask, _rng: &mut Rng) -> Action {
        let s = observation.state_index.expect("a chase observation") as usize;
        let (from, target) = (s / self.cells as usize, s % self.cells as usize);
        // With nowhere to go it pushes north against whatever is there, which leaves it in place.
        self.first_move(from, target).unwrap_or(Direction::North).action()
    }

    fn label(&self) -> &str {
        "chase_shortest_path"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_chaser_goes_around_walls() {
        // .#.
        // ...
        let walls = [false, true, false, false, false, false];
        let chase = ChaseShortestPath::new(3, 2, &walls);
        // From the top-left to the top-right: down, not into the wall.
        assert_eq!(chase.first_move(0, 2), Some(Direction::South));
        assert_eq!(chase.first_move(3, 2), Some(Direction::East));
        assert_eq!(chase.first_move(2, 2), None, "already there");
        let mut p = &chase;
        let a = p.act(&chase.observation(0, 2), &ActionMask::all(4), &mut Rng::new(1));
        assert_eq!(a, Direction::South.action());
    }
}
