//! Web client for Reinforcement Odyssey.
//!
//! - `content`: level and tutorial files compiled into the app.
//! - `campaign`: unlocks and best results, saved in the browser.
//! - `protocol` and `sim`: the simulation, run inside a Web Worker. `sim` is
//!   plain Rust and is tested natively.
//! - `target`: where training runs (this browser for now).
//! - `experiments`: the player's experiment log, for comparing runs.
//! - `library`: the concept library.
//! - `store`: trained rafts saved in IndexedDB.
//! - `render`: the phosphor Canvas2D renderer. It reads only render models.
//! - `walkthrough`: the live Q-learning pseudocode.
//! - `app`: the Leptos UI shell.

pub mod campaign;
pub mod chart_math;
pub mod content;
pub mod experiments;
pub mod lesson_code;
pub mod protocol;
pub mod sim;
pub mod target;

#[cfg(target_arch = "wasm32")]
pub mod analyse;
#[cfg(target_arch = "wasm32")]
pub mod app;
#[cfg(target_arch = "wasm32")]
pub mod compare;
#[cfg(target_arch = "wasm32")]
pub mod library;
#[cfg(target_arch = "wasm32")]
pub mod render;
#[cfg(target_arch = "wasm32")]
pub mod store;
#[cfg(target_arch = "wasm32")]
pub mod training_ui;
#[cfg(target_arch = "wasm32")]
pub mod walkthrough;
#[cfg(target_arch = "wasm32")]
pub mod worker;
