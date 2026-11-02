//! Observations produced from an `ObservationSpec`.

use serde::{Deserialize, Serialize};

/// What one agent observes at one step.
///
/// `features` is always filled, in spec order, ready for neural networks.
/// `state_index` is filled when every feature is discrete; tabular agents use it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub features: Vec<f32>,
    pub state_index: Option<u32>,
}
