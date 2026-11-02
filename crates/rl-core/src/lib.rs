//! Core abstractions shared by every engine, agent, trainer and front end.
//!
//! `rl-core` must stay independent of graphics, UI, networking and ML
//! frameworks. See `docs/environment-model.md`.

pub mod action;
pub mod observation;
pub mod policy;
pub mod render;
pub mod rng;
pub mod spec;
pub mod world;

pub use action::{Action, ActionMask, JointAction};
pub use observation::Observation;
pub use policy::PolicyProvider;
pub use render::{Glyph, RenderAgent, RenderEvent, RenderModel};
pub use rng::Rng;
pub use spec::{ActionSpace, AgentSpec, FeatureSpec, ObservationSpec, SpecHash, WorldSpec};
pub use world::{AgentId, StepEvent, StepInfo, StepResult, StepResults, World};
