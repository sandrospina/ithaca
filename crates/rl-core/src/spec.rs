//! Observation, action and world specifications.
//!
//! A trained policy stores the `SpecHash` of the observation and action specs
//! it was trained on, and refuses to load into a world with different specs.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionSpace {
    /// `n` discrete actions, numbered `0..n`.
    Discrete { n: u32, labels: Vec<String> },
}

impl ActionSpace {
    pub fn discrete_count(&self) -> u32 {
        match self {
            ActionSpace::Discrete { n, .. } => *n,
        }
    }
}

/// One named feature in an observation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureSpec {
    /// Canonical description, e.g. `self_position(encoding: index)`.
    pub name: String,
    /// Number of floats this feature contributes.
    pub size: u32,
    /// Number of distinct values if the feature is discrete, used for tabular state indices.
    pub cardinality: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationSpec {
    pub features: Vec<FeatureSpec>,
}

impl ObservationSpec {
    pub fn size(&self) -> u32 {
        self.features.iter().map(|f| f.size).sum()
    }

    /// Number of discrete states, if every feature is discrete. This is the
    /// Q-table row count shown to the player.
    pub fn state_count(&self) -> Option<u64> {
        self.features.iter().try_fold(1u64, |acc, f| {
            f.cardinality.map(|c| acc.saturating_mul(u64::from(c)))
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentSpec {
    pub name: String,
    pub observation: ObservationSpec,
    pub action: ActionSpace,
}

impl AgentSpec {
    /// Stable hash of the observation and action specs.
    pub fn spec_hash(&self) -> SpecHash {
        let mut h = Fnv1a::new();
        for f in &self.observation.features {
            h.write(f.name.as_bytes());
            h.write(&f.size.to_le_bytes());
            h.write(&f.cardinality.unwrap_or(0).to_le_bytes());
        }
        h.write(b"|");
        match &self.action {
            ActionSpace::Discrete { n, .. } => {
                h.write(b"discrete");
                h.write(&n.to_le_bytes());
            }
        }
        SpecHash(h.finish())
    }
}

/// Specs for every externally controlled agent, indexed by `AgentId`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldSpec {
    pub agents: Vec<AgentSpec>,
}

/// Stable 64-bit spec fingerprint (FNV-1a). Unlike `std` hashers, the value
/// never changes between Rust versions or platforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpecHash(pub u64);

struct Fnv1a(u64);

impl Fnv1a {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent(features: Vec<FeatureSpec>) -> AgentSpec {
        AgentSpec {
            name: "odysseus".into(),
            observation: ObservationSpec { features },
            action: ActionSpace::Discrete {
                n: 4,
                labels: vec![],
            },
        }
    }

    fn pos() -> FeatureSpec {
        FeatureSpec {
            name: "self_position(encoding: index)".into(),
            size: 1,
            cardinality: Some(25),
        }
    }

    #[test]
    fn state_count_multiplies_cardinalities() {
        let a = agent(vec![pos(), pos()]);
        assert_eq!(a.observation.state_count(), Some(625));
    }

    #[test]
    fn spec_hash_detects_changes() {
        let a = agent(vec![pos()]);
        let b = agent(vec![pos(), pos()]);
        assert_eq!(a.spec_hash(), agent(vec![pos()]).spec_hash());
        assert_ne!(a.spec_hash(), b.spec_hash());
    }
}
