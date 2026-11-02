//! Actions and action masks.

use serde::{Deserialize, Serialize};

/// One agent's action. Continuous actions arrive with the Sea engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    Discrete(u32),
}

impl Action {
    pub fn discrete(self) -> u32 {
        match self {
            Action::Discrete(a) => a,
        }
    }
}

/// Actions for every externally controlled agent, indexed by `AgentId`.
pub type JointAction = Vec<Action>;

/// Fixed-size mask of valid discrete actions (up to 64). Never allocates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionMask {
    bits: u64,
    len: u8,
}

impl ActionMask {
    /// A mask where all `len` actions are valid.
    pub fn all(len: u32) -> Self {
        assert!((1..=64).contains(&len), "action count must be 1..=64");
        let bits = if len == 64 {
            u64::MAX
        } else {
            (1u64 << len) - 1
        };
        Self {
            bits,
            len: len as u8,
        }
    }

    pub fn len(&self) -> u32 {
        u32::from(self.len)
    }

    pub fn is_empty(&self) -> bool {
        self.bits == 0
    }

    pub fn is_valid(&self, action: u32) -> bool {
        action < self.len() && self.bits & (1 << action) != 0
    }

    pub fn set(&mut self, action: u32, valid: bool) {
        assert!(action < self.len());
        if valid {
            self.bits |= 1 << action;
        } else {
            self.bits &= !(1 << action);
        }
    }

    pub fn count_valid(&self) -> u32 {
        self.bits.count_ones()
    }

    /// Iterates over valid action indices in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.len()).filter(|a| self.is_valid(*a))
    }

    /// The `n`-th valid action in ascending order.
    pub fn nth_valid(&self, n: u32) -> Option<u32> {
        self.iter().nth(n as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_basics() {
        let mut m = ActionMask::all(4);
        assert_eq!(m.count_valid(), 4);
        m.set(2, false);
        assert!(!m.is_valid(2));
        assert!(!m.is_valid(4));
        assert_eq!(m.iter().collect::<Vec<_>>(), vec![0, 1, 3]);
        assert_eq!(m.nth_valid(2), Some(3));
    }
}
