//! Web client for Reinforcement Odyssey.
//!
//! - `content`: level and tutorial files compiled into the app.
//! - `campaign`: unlocks and best results, saved in the browser.
//! - `protocol` and `sim`: the simulation, run inside a Web Worker. `sim` is
//!   plain Rust and is tested natively.
//! - `render`: the phosphor Canvas2D renderer. It reads only render models.
//! - `app`: the Leptos UI shell.

pub mod campaign;
pub mod content;
pub mod protocol;
pub mod sim;

#[cfg(target_arch = "wasm32")]
pub mod app;
#[cfg(target_arch = "wasm32")]
pub mod render;
#[cfg(target_arch = "wasm32")]
pub mod worker;
