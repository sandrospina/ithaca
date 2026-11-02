//! The Leptos UI: amber terminal shell around the phosphor arena.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

use gloo_worker::Spawnable;
use leptos::ev;
use leptos::prelude::*;
use rl_tutorial::{Beat, Depth, StepDef, TutorialDef, UiElement};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::Closure;

use crate::campaign::{Progress, STORAGE_KEY};
use crate::content::{self, LEVELS};
use crate::protocol::{ObjectiveGoal, Outcome, Pilot, SailedBy, SimRequest, SimResponse, Snapshot};
use crate::render::{self, Effects, Highlights, RenderState};
use crate::worker::SimWorker;

/// Pause between agent steps when watching an agent sail, in milliseconds.
const AUTOPILOT_MS: u64 = 90;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum View {
    Arena,
    StyleSheet,
}

/// Optional URL parameters, used for debugging and automated screenshots:
/// `?level=l01_the_raft&seed=3&moves=EESS&step=2&view=stylesheet&plain=1&unlock=all&watch=1`.
struct UrlParams {
    level: Option<String>,
    seed: Option<u64>,
    view: View,
    plain: bool,
    moves: Vec<u32>,
    tutorial_step: Option<usize>,
    unlock_all: bool,
    watch: bool,
}

impl UrlParams {
    fn read() -> Self {
        let search = window().location().search().unwrap_or_default();
        let p = web_sys::UrlSearchParams::new_with_str(&search).ok();
        let get = |k: &str| p.as_ref().and_then(|p| p.get(k));
        Self {
            level: get("level"),
            seed: get("seed").and_then(|s| s.parse().ok()),
            view: if get("view").as_deref() == Some("stylesheet") {
                View::StyleSheet
            } else {
                View::Arena
            },
            plain: get("plain").as_deref() == Some("1"),
            moves: get("moves")
                .unwrap_or_default()
                .chars()
                .filter_map(|c| match c.to_ascii_uppercase() {
                    'N' => Some(0),
                    'E' => Some(1),
                    'S' => Some(2),
                    'W' => Some(3),
                    _ => None,
                })
                .collect(),
            tutorial_step: get("step").and_then(|s| s.parse().ok()),
            unlock_all: get("unlock").as_deref() == Some("all"),
            watch: get("watch").as_deref() == Some("1"),
        }
    }
}

fn now_ms() -> f64 {
    window().performance().map_or(0.0, |p| p.now())
}

fn load_progress() -> Progress {
    window()
        .local_storage()
        .ok()
        .flatten()
        .and_then(|s| s.get_item(STORAGE_KEY).ok().flatten())
        .map(|json| Progress::from_json(&json))
        .unwrap_or_default()
}

fn save_progress(progress: &Progress) {
    if let Some(storage) = window().local_storage().ok().flatten() {
        let _ = storage.set_item(STORAGE_KEY, &progress.to_json());
    }
}

fn level_index(id: &str) -> Option<usize> {
    LEVELS.iter().position(|l| l.id == id)
}

type FrameCallback = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;

/// Calls `frame` on every animation frame, forever.
fn animation_loop(mut frame: impl FnMut(f64) + 'static) {
    let slot: FrameCallback = Rc::new(RefCell::new(None));
    let again = slot.clone();
    *slot.borrow_mut() = Some(Closure::new(move |t: f64| {
        frame(t);
        if let Some(cb) = again.borrow().as_ref() {
            let _ = window().request_animation_frame(cb.as_ref().unchecked_ref());
        }
    }));
    if let Some(cb) = slot.borrow().as_ref() {
        let _ = window().request_animation_frame(cb.as_ref().unchecked_ref());
    }
}

/// The numbers next to the objective: the target, this voyage, and the
/// player's record or the agent baseline.
fn objective_stats(s: &Snapshot, progress: &Progress) -> String {
    let Some(objective) = &s.objective else {
        return String::new();
    };
    match objective.goal {
        ObjectiveGoal::BestReturn { target, moves } => {
            let best = progress
                .best(&s.level_id)
                .map_or("—".into(), |b| format!("{b:+}"));
            format!(
                "best possible {target:+} in {moves} moves · this voyage {:+} after {} moves · your best {best}",
                s.episode_return, s.step
            )
        }
        ObjectiveGoal::SuccessRate { .. } => {
            let mut parts = Vec::new();
            if let Some(prev) = level_index(&s.level_id)
                .and_then(|i| i.checked_sub(1))
                .map(|i| &LEVELS[i])
                && let Some(best) = progress.best(prev.id)
            {
                parts.push(format!(
                    "your best by hand in {}: {best:+}",
                    prev.menu_label
                ));
            }
            if let Some(b) = s.random_baseline {
                parts.push(format!(
                    "random agent over {} voyages: reaches the shore {:.0}% · mean return {:+.1}",
                    b.episodes,
                    b.success_rate * 100.0,
                    b.mean_return
                ));
            }
            parts.push("TRAIN unlocks with Q-learning in Milestone 1".into());
            parts.join(" · ")
        }
    }
}

#[component]
pub fn App() -> impl IntoView {
    let params = UrlParams::read();
    let unlock_all = params.unlock_all;
    let progress = RwSignal::new(load_progress());
    let can_open = move |i: usize| unlock_all || progress.with(|p| p.is_unlocked(i));

    let initial = params
        .level
        .as_deref()
        .and_then(level_index)
        .filter(|&i| unlock_all || progress.with_untracked(|p| p.is_unlocked(i)))
        .unwrap_or_else(|| progress.with_untracked(|p| p.resume_index()));
    let level_id = RwSignal::new(LEVELS[initial].id.to_string());

    let snapshot = RwSignal::new(None::<Snapshot>);
    let error = RwSignal::new(None::<String>);
    let seed = RwSignal::new(params.seed.unwrap_or(1));
    let view = RwSignal::new(params.view);
    let plain = RwSignal::new(params.plain);
    let depth = RwSignal::new(Depth::Intuition);
    let tutorial_index = RwSignal::new(0usize);
    let just_completed = RwSignal::new(false);
    let pending_moves = StoredValue::new(Some(params.moves));
    let pending_step = StoredValue::new(params.tutorial_step);
    let pending_watch = StoredValue::new(params.watch);

    // ---- simulation worker
    let bridge = SimWorker::spawner()
        .callback(move |response| match response {
            SimResponse::Snapshot(s) => {
                error.set(None);
                snapshot.set(Some(*s));
            }
            SimResponse::Error(e) => error.set(Some(e)),
        })
        // Trunk builds the worker as a classic (non-module) script.
        .as_module(false)
        .spawn("./worker.js");
    let bridge = StoredValue::new_local(bridge);
    let send = move |request: SimRequest| bridge.with_value(|b| b.send(request));

    // ---- autopilot: an agent sails while the player watches
    let autopilot = StoredValue::new_local(None::<IntervalHandle>);
    let autopilot_on = RwSignal::new(false);
    let stop_autopilot = move || {
        autopilot.update_value(|h| {
            if let Some(h) = h.take() {
                h.clear();
            }
        });
        autopilot_on.set(false);
    };
    let start_autopilot = move || {
        stop_autopilot();
        send(SimRequest::Reset {
            seed: seed.get_untracked(),
        });
        if let Ok(h) = set_interval_with_handle(
            move || send(SimRequest::AgentStep),
            Duration::from_millis(AUTOPILOT_MS),
        ) {
            autopilot.set_value(Some(h));
            autopilot_on.set(true);
        }
    };

    // ---- tutorial
    let tutorial = Memo::new(move |_| {
        content::level(&level_id.get()).and_then(|e| TutorialDef::parse(e.tutorial_source).ok())
    });
    let highlights = Memo::new(move |_| {
        let i = tutorial_index.get();
        tutorial.with(|t| {
            t.as_ref()
                .and_then(|t| t.steps.get(i))
                .map(|StepDef::Say { highlight, .. }| highlight.clone())
                .unwrap_or_default()
        })
    });
    let is_hl = move |el: UiElement| highlights.with(|h| h.contains(&el));
    let step_count = move || tutorial.with(|t| t.as_ref().map_or(0, |t| t.steps.len()));
    let current_step = move || {
        let i = tutorial_index.get();
        let d = depth.get();
        tutorial.with(|t| {
            t.as_ref().and_then(|t| t.steps.get(i)).map(
                |StepDef::Say {
                     beat, title, text, ..
                 }| {
                    (
                        beat.label().to_string(),
                        title.clone(),
                        text.at(d).to_string(),
                    )
                },
            )
        })
    };

    // ---- level loading
    Effect::new(move |_| {
        let id = level_id.get();
        stop_autopilot();
        just_completed.set(false);
        tutorial_index.set(
            pending_step
                .try_update_value(|s| s.take())
                .flatten()
                .unwrap_or(0),
        );
        send(SimRequest::Load {
            level_id: id,
            seed: seed.get_untracked(),
        });
        for action in pending_moves
            .try_update_value(|m| m.take())
            .flatten()
            .unwrap_or_default()
        {
            send(SimRequest::Step { action });
        }
        if pending_watch
            .try_update_value(std::mem::take)
            .unwrap_or(false)
        {
            start_autopilot();
        }
    });

    // ---- campaign progress
    Effect::new(move |_| progress.with(save_progress));
    let recorded = StoredValue::new(None::<(String, u32)>);
    Effect::new(move |_| {
        let Some(s) = snapshot.get() else { return };
        if s.outcome == Outcome::Sailing {
            return;
        }
        if autopilot_on.get_untracked() {
            stop_autopilot();
        }
        let key = (s.level_id.clone(), s.episode);
        if recorded.get_value().as_ref() == Some(&key) {
            return;
        }
        recorded.set_value(Some(key));
        let mut newly_completed = false;
        progress.update(|p| newly_completed = p.record(&s));
        if newly_completed {
            just_completed.set(true);
            let debrief = tutorial.with_untracked(|t| {
                t.as_ref()
                    .and_then(|t| t.steps.iter().position(|s| s.beat() == Beat::Debrief))
            });
            if let Some(i) = debrief {
                tutorial_index.set(i);
            }
        }
    });
    let level_complete = move || progress.with(|p| p.is_complete(&level_id.get()));
    let next_level = move || {
        level_index(&level_id.get())
            .map(|i| i + 1)
            .filter(|&i| i < LEVELS.len())
    };
    let pilot = move || snapshot.with(|s| s.as_ref().map(|s| s.pilot));

    // ---- keyboard
    let _keys = window_event_listener(ev::keydown, move |e: web_sys::KeyboardEvent| {
        if let Some(el) = e
            .target()
            .and_then(|t| t.dyn_into::<web_sys::Element>().ok())
            && matches!(el.tag_name().as_str(), "INPUT" | "SELECT" | "TEXTAREA")
        {
            return;
        }
        let action = match e.key().as_str() {
            "ArrowUp" | "w" | "W" => Some(0),
            "ArrowRight" | "d" | "D" => Some(1),
            "ArrowDown" | "s" | "S" => Some(2),
            "ArrowLeft" | "a" | "A" => Some(3),
            _ => None,
        };
        if let Some(action) = action {
            e.prevent_default();
            send(SimRequest::Step { action });
            return;
        }
        match e.key().as_str() {
            "r" | "R" => {
                stop_autopilot();
                send(SimRequest::Reset {
                    seed: seed.get_untracked(),
                });
            }
            "n" | "N" => {
                stop_autopilot();
                let s = seed.get_untracked().wrapping_add(1);
                seed.set(s);
                send(SimRequest::Reset { seed: s });
            }
            _ => {}
        }
    });

    // ---- rendering
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let render_state = StoredValue::new_local(RenderState::default());
    let started = StoredValue::new(false);

    Effect::new(move |_| {
        snapshot.with(|s| {
            if let Some(s) = s {
                let now = now_ms();
                render_state.update_value(|rs| rs.on_snapshot(&s.render, now, s.step == 0));
            }
        });
    });

    Effect::new(move |_| {
        let Some(canvas) = canvas_ref.get() else {
            return;
        };
        if started.get_value() {
            return;
        }
        started.set_value(true);
        let Some(ctx) = canvas
            .get_context("2d")
            .ok()
            .flatten()
            .and_then(|c| c.dyn_into::<web_sys::CanvasRenderingContext2d>().ok())
        else {
            error.set(Some("this browser does not support Canvas2D".into()));
            return;
        };
        animation_loop(move |now| {
            let fx = Effects {
                plain: plain.get_untracked(),
            };
            match view.get_untracked() {
                View::StyleSheet => render::draw_style_sheet(&ctx, now, fx),
                View::Arena => snapshot.with_untracked(|s| {
                    if let Some(s) = s {
                        let hl = highlights.with_untracked(|h| Highlights {
                            agent: h.contains(&UiElement::Agent),
                            goal: h.contains(&UiElement::Goal),
                        });
                        render_state.update_value(|rs| {
                            render::draw_world(&ctx, &s.render, rs, now, hl, fx)
                        });
                    }
                }),
            }
        });
    });

    Effect::new(move |_| {
        let on = plain.get();
        if let Some(body) = document().body() {
            let _ = body.class_list().toggle_with_force("plain", on);
        }
    });

    // ---- view helpers
    let snap_text = move |f: fn(&Snapshot) -> String| {
        move || snapshot.with(|s| s.as_ref().map_or("—".into(), f))
    };

    view! {
        <div class="app">
            <header class="topbar">
                <span class="title glow">"REINFORCEMENT ODYSSEY"</span>
                <span class="subtitle">"THE ROAD TO ITHACA"</span>
                <span style="flex: 1"></span>
                <button class:active=move || view.get() == View::Arena on:click=move |_| view.set(View::Arena)>
                    "ARENA"
                </button>
                <button
                    class:active=move || view.get() == View::StyleSheet
                    on:click=move |_| view.set(View::StyleSheet)
                >
                    "STYLE SHEET"
                </button>
                <button class:active=move || plain.get() on:click=move |_| plain.update(|p| *p = !*p)>
                    "PLAIN MODE"
                </button>
            </header>

            <nav class="voyage" aria-label="Voyage map">
                <span class="port">"OGYGIA"</span>
                {LEVELS
                    .iter()
                    .enumerate()
                    .map(|(i, l)| {
                        let done = move || progress.with(|p| p.is_complete(l.id));
                        view! {
                            <span class="route">"━━"</span>
                            <button
                                class="stop"
                                class:active=move || level_id.get() == l.id
                                class:done=done
                                disabled=move || !can_open(i)
                                title=move || if can_open(i) { "" } else { "Complete the previous level to unlock" }
                                on:click=move |_| level_id.set(l.id.to_string())
                            >
                                {move || if done() { "✓ " } else if can_open(i) { "○ " } else { "■ " }}
                                {l.menu_label}
                            </button>
                        }
                    })
                    .collect_view()}
                <span class="route">"━━ ┄ ┄ ┄"</span>
                <span class="port far">"ITHACA"</span>
            </nav>

            <section class="objective" class:hl-box=move || is_hl(UiElement::Objective)>
                <span class="objective-label">"OBJECTIVE"</span>
                <span class="objective-text">
                    {move || {
                        snapshot
                            .with(|s| s.as_ref().and_then(|s| s.objective.as_ref().map(|o| o.text.clone())))
                            .unwrap_or_default()
                    }}
                </span>
                <span class="objective-stats">
                    {move || {
                        snapshot.with(|s| s.as_ref().map(|s| progress.with(|p| objective_stats(s, p)))).unwrap_or_default()
                    }}
                </span>
                {move || {
                    level_complete()
                        .then(|| {
                            view! {
                                <span class="complete glow" class:fresh=move || just_completed.get()>
                                    "✓ LEVEL COMPLETE"
                                </span>
                            }
                        })
                }}
                {move || {
                    next_level()
                        .filter(|&i| level_complete() && can_open(i))
                        .map(|i| {
                            view! {
                                <button class="sail-on" on:click=move |_| level_id.set(LEVELS[i].id.to_string())>
                                    {format!("SAIL ON TO {} ▶", LEVELS[i].menu_label)}
                                </button>
                            }
                        })
                }}
            </section>

            <main class="main">
                <section class="arena-wrap" class:hl-box=move || is_hl(UiElement::Arena)>
                    <canvas node_ref=canvas_ref width="720" height="720" aria-label="Arena"></canvas>
                </section>

                <aside class="side">
                    <section class="panel">
                        <div class="beat">{move || current_step().map(|s| s.0)}</div>
                        <h2 class="glow">{move || current_step().map(|s| s.1)}</h2>
                        <div class="tutorial-text">{move || current_step().map(|s| s.2)}</div>
                        <div class="row">
                            {[(Depth::Story, "STORY"), (Depth::Intuition, "INTUITION"), (Depth::Math, "MATH")]
                                .into_iter()
                                .map(|(d, label)| {
                                    view! {
                                        <button class:active=move || depth.get() == d on:click=move |_| depth.set(d)>
                                            {label}
                                        </button>
                                    }
                                })
                                .collect_view()}
                        </div>
                        <div class="row" style="margin-top: 0.5rem">
                            <button
                                disabled=move || tutorial_index.get() == 0
                                on:click=move |_| tutorial_index.update(|i| *i = i.saturating_sub(1))
                            >
                                "◀ PREV"
                            </button>
                            <button
                                disabled=move || { tutorial_index.get() + 1 >= step_count() }
                                on:click=move |_| tutorial_index.update(|i| *i += 1)
                            >
                                "NEXT ▶"
                            </button>
                            <span class="subtitle">
                                {move || format!("{} / {}", tutorial_index.get() + 1, step_count())}
                            </span>
                        </div>
                    </section>

                    <section class="panel">
                        <h2>"THE RL LOOP"</h2>
                        <div class="hud">
                            <span class="label" class:hl=move || is_hl(UiElement::State)>"STATE"</span>
                            <span class="value">
                                {snap_text(|s| match (s.state_index, s.state_count) {
                                    (Some(i), Some(n)) => format!("cell {i} · ({}, {}) · {n} states", s.position.0, s.position.1),
                                    _ => format!("({}, {})", s.position.0, s.position.1),
                                })}
                            </span>
                            <span class="label" class:hl=move || is_hl(UiElement::Action)>"ACTION"</span>
                            <span class="value">{snap_text(|s| s.last_action.clone().unwrap_or("—".into()))}</span>
                            <span class="label" class:hl=move || is_hl(UiElement::Reward)>"REWARD"</span>
                            <span class="value">
                                {snap_text(|s| s.last_reward.map_or("—".into(), |r| format!("{r:+}")))}
                            </span>
                            <span class="label" class:hl=move || is_hl(UiElement::Return)>"RETURN"</span>
                            <span class="value">{snap_text(|s| format!("{:+}", s.episode_return))}</span>
                            <span class="label" class:hl=move || is_hl(UiElement::Step)>"STEP"</span>
                            <span class="value">{snap_text(|s| format!("{} / {}", s.step, s.max_steps))}</span>
                            <span class="label" class:hl=move || is_hl(UiElement::Episode)>"EPISODE"</span>
                            <span class="value">
                                {snap_text(|s| {
                                    let status = match s.outcome {
                                        Outcome::Sailing => "sailing",
                                        Outcome::ReachedShore => "reached the shore",
                                        Outcome::OutOfTime => "out of time",
                                    };
                                    let pilot = match s.sailed_by {
                                        SailedBy::Nobody => "",
                                        SailedBy::Player => " · by you",
                                        SailedBy::RandomAgent => " · by the random agent",
                                    };
                                    format!("{} · {status}{pilot}", s.episode)
                                })}
                            </span>
                        </div>
                    </section>

                    <section class="panel">
                        <h2>"VOYAGE"</h2>
                        <div class="row" class:hl-box=move || is_hl(UiElement::Seed)>
                            "SEED"
                            <input
                                type="number"
                                min="0"
                                prop:value=move || seed.get().to_string()
                                on:change=move |e| {
                                    if let Ok(s) = event_target_value(&e).parse::<u64>() {
                                        stop_autopilot();
                                        seed.set(s);
                                        send(SimRequest::Reset { seed: s });
                                    }
                                }
                            />
                            <button on:click=move |_| {
                                stop_autopilot();
                                send(SimRequest::Reset { seed: seed.get_untracked() });
                            }>"RESET (R)"</button>
                            <button on:click=move |_| {
                                stop_autopilot();
                                let s = seed.get_untracked().wrapping_add(1);
                                seed.set(s);
                                send(SimRequest::Reset { seed: s });
                            }>"NEW SEED (N)"</button>
                        </div>
                        {move || match pilot() {
                            Some(Pilot::Agent) => {
                                view! {
                                    <div class="row pilot" class:hl-box=move || is_hl(UiElement::Autopilot)>
                                        <button
                                            class="primary"
                                            on:click=move |_| {
                                                if autopilot_on.get_untracked() { stop_autopilot() } else { start_autopilot() }
                                            }
                                        >
                                            {move || if autopilot_on.get() { "■ STOP" } else { "▶ WATCH RANDOM AGENT" }}
                                        </button>
                                        <span class="subtitle">"steering is off · the agent sails"</span>
                                    </div>
                                }
                                    .into_any()
                            }
                            _ => {
                                view! {
                                    <p class="row pilot" class:hl-box=move || is_hl(UiElement::Controls)>
                                        "Steer with the arrow keys or W A S D."
                                    </p>
                                }
                                    .into_any()
                            }
                        }}
                        <div class="log">
                            {move || {
                                snapshot
                                    .with(|s| s.as_ref().map(|s| s.log.clone()).unwrap_or_default())
                                    .into_iter()
                                    .map(|line| view! { <div class="new">{"> "}{line}</div> })
                                    .collect_view()
                            }}
                        </div>
                        {move || error.get().map(|e| view! { <p style="color: var(--adversary)">{"ERROR: "}{e}</p> })}
                    </section>
                </aside>
            </main>

            <nav class="phasebar">
                {["CONFIGURE", "TRAIN", "ANALYSE", "DEPLOY"]
                    .into_iter()
                    .map(|p| view! { <button class="phase" disabled=true title="Unlocks with Q-learning in Milestone 1">{p}</button> })
                    .collect_view()}
            </nav>
            <footer class="statusbar">
                <span>"TRAINING ON: local · simulation in Web Worker"</span>
                <span style="flex: 1"></span>
                <button
                    class="quiet"
                    on:click=move |_| {
                        if window().confirm_with_message("Reset all campaign progress?").unwrap_or(false) {
                            progress.set(Progress::default());
                            level_id.set(LEVELS[0].id.to_string());
                        }
                    }
                >
                    "RESET PROGRESS"
                </button>
                <span>"MILESTONE 0"</span>
            </footer>
        </div>
    }
}
