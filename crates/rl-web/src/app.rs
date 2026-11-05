//! The Leptos UI: amber terminal shell around the phosphor arena.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::time::Duration;

use gloo_worker::Spawnable;
use leptos::ev;
use leptos::prelude::*;
use rl_tutorial::{Beat, Depth, StepDef, TutorialDef, UiElement};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::Closure;

use crate::analyse::{AnalyseCtx, analyse_view};
use crate::campaign::{Answer, AnswerOutcome, Progress, STORAGE_KEY};
use crate::chart_math::signed;
use crate::compare::{CompareCtx, Pending, compare_view};
use crate::content::{self, LEVELS};
use crate::experiments::{ExperimentLog, Selection};
use crate::lesson_code::{self, LineStep, Reveal, UPDATE};
use crate::library::{LibraryCtx, library_view};
use crate::protocol::{
    BatchReport, DeployReport, ExperimentResult, GoalDef, LessonMode, LessonState, Outcome, Pilot, PlayerEdits,
    QLearningConfig, SailedBy, SavedRaft, SimRequest, SimResponse, Snapshot,
};
use crate::render::{self, Effects, Highlights, Layers, Marks, RenderState, Updated};
use crate::sim::experiment_spec;
use crate::store;
use crate::target::{LocalWorker, TrainingTarget};
use crate::training_ui::{Draft, TrainCtx, configure_view, deploy_view, train_view};
use crate::walkthrough::{Play, WalkCtx, walkthrough_view};
use crate::worker::SimWorker;

/// Pause between agent steps when watching an agent sail, in milliseconds.
const AUTOPILOT_MS: u64 = 90;

/// The phases come in two stages: measuring the untrained agent, then learning.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// ANALYSE.
    Measure,
    /// CONFIGURE, TRAIN and DEPLOY.
    Learn,
}

impl Stage {
    /// The stage a tutorial highlight points into.
    fn of(el: UiElement) -> Option<Stage> {
        match el {
            UiElement::Analyse | UiElement::BestSeed | UiElement::Quiz => Some(Stage::Measure),
            UiElement::Configure
            | UiElement::Rewards
            | UiElement::Train
            | UiElement::Walkthrough
            | UiElement::Deploy
            | UiElement::Compare
            | UiElement::Observation => Some(Stage::Learn),
            _ => None,
        }
    }

    fn of_view(v: View) -> Stage {
        if v == View::Analyse { Stage::Measure } else { Stage::Learn }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum View {
    Arena,
    StyleSheet,
    Configure,
    Train,
    /// The live Q-learning walkthrough: the arena plus the pseudocode.
    Walkthrough,
    Analyse,
    Deploy,
    /// Experiments and their comparison.
    Compare,
    /// The concept library.
    Library,
}

/// The side panels below the briefing, which can be folded to a one-line summary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SidePanel {
    Loop,
    QValues,
    Voyage,
}

impl SidePanel {
    const ALL: [SidePanel; 3] = [SidePanel::Loop, SidePanel::QValues, SidePanel::Voyage];

    /// Folded unless the view needs it: the walkthrough shows the RL loop in
    /// its own code panel, and the panel views have no arena to describe.
    fn folded_in(self, view: View) -> bool {
        match view {
            View::Arena | View::StyleSheet => false,
            View::Walkthrough => self != SidePanel::QValues,
            _ => true,
        }
    }

    /// Tutorial highlights that live inside this panel and unfold it.
    fn elements(self) -> &'static [UiElement] {
        match self {
            SidePanel::Loop => &[
                UiElement::State,
                UiElement::Action,
                UiElement::Reward,
                UiElement::Return,
                UiElement::Step,
                UiElement::Episode,
            ],
            SidePanel::QValues => &[UiElement::QValues],
            SidePanel::Voyage => &[UiElement::Seed, UiElement::Autopilot, UiElement::Controls],
        }
    }
}

/// A side panel whose body folds away, leaving its title and a one-line summary.
#[component]
fn Fold(
    title: &'static str,
    #[prop(into)] folded: Signal<bool>,
    on_toggle: Callback<()>,
    #[prop(into)] summary: Signal<String>,
    #[prop(into)] hl: Signal<bool>,
    children: Children,
) -> impl IntoView {
    view! {
        <section class="panel" class:folded=move || folded.get() class:hl-box=move || hl.get()>
            <button
                class="panel-head"
                aria-expanded=move || (!folded.get()).to_string()
                title=move || if folded.get() { "Show" } else { "Fold away" }
                on:click=move |_| on_toggle.run(())
            >
                <span class="panel-title">{title}</span>
                <span class="fold-summary">{move || if folded.get() { summary.get() } else { String::new() }}</span>
                <span class="fold-mark">{move || if folded.get() { "▸" } else { "▾" }}</span>
            </button>
            <div class="panel-body" style:display=move || if folded.get() { "none" } else { "block" }>
                {children()}
            </div>
        </section>
    }
}

impl View {
    /// Views that replace the arena with a panel.
    fn is_panel(self) -> bool {
        matches!(self, View::Configure | View::Train | View::Analyse | View::Deploy | View::Compare | View::Library)
    }
}

/// Episodes trained per request while a training run is in progress.
const TRAIN_CHUNK: u32 = 50;

/// Optional URL parameters, used for debugging and automated screenshots:
/// `?level=l01_the_raft&seed=3&moves=EESS&step=2&view=stylesheet&plain=1&unlock=all&watch=1&batch=100&rewards=lure:5&train=800&deploy=1`.
/// `view` also accepts `configure`, `train`, `walkthrough`, `analyse` and `deploy`.
struct UrlParams {
    level: Option<String>,
    seed: Option<u64>,
    view: View,
    plain: bool,
    moves: Vec<u32>,
    tutorial_step: Option<usize>,
    unlock_all: bool,
    watch: bool,
    batch: Option<u32>,
    /// Reward overrides applied to a new raft on load, e.g. `lure:5,hazard:-30`.
    rewards: Option<Vec<(String, f32)>>,
    /// Episodes to train on load.
    train: Option<u32>,
    /// Deploy once the training on load finishes.
    deploy: bool,
}

impl UrlParams {
    fn read() -> Self {
        let search = window().location().search().unwrap_or_default();
        let p = web_sys::UrlSearchParams::new_with_str(&search).ok();
        let get = |k: &str| p.as_ref().and_then(|p| p.get(k));
        Self {
            level: get("level"),
            seed: get("seed").and_then(|s| s.parse().ok()),
            view: match get("view").as_deref() {
                Some("stylesheet") => View::StyleSheet,
                Some("analyse") => View::Analyse,
                Some("configure") => View::Configure,
                Some("train") => View::Train,
                Some("walkthrough") => View::Walkthrough,
                Some("deploy") => View::Deploy,
                Some("compare") => View::Compare,
                Some("library") => View::Library,
                _ => View::Arena,
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
            batch: get("batch").and_then(|b| b.parse().ok()),
            rewards: get("rewards").map(|r| {
                r.split(',')
                    .filter_map(|kv| {
                        let (k, v) = kv.split_once(':')?;
                        Some((k.trim().to_string(), v.trim().parse().ok()?))
                    })
                    .collect()
            }),
            train: get("train").and_then(|t| t.parse().ok()),
            deploy: get("deploy").as_deref() == Some("1"),
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
    match (&objective.goal, objective.best_route) {
        (GoalDef::BestReturn, Some(best)) => {
            let mine = progress.best(&s.level_id).map_or("—".into(), |b| signed(f64::from(b)));
            format!(
                "best possible {} in {} moves · this voyage {} after {} moves · your best {mine}",
                signed(f64::from(best.episode_return)),
                best.moves,
                signed(f64::from(s.episode_return)),
                s.step
            )
        }
        _ => {
            let mut parts = Vec::new();
            if let Some(prev) = level_index(&s.level_id).and_then(|i| i.checked_sub(1)).map(|i| &LEVELS[i])
                && let Some(best) = progress.best(prev.id)
            {
                parts.push(format!("your best by hand in {}: {}", prev.menu_label, signed(f64::from(best))));
            }
            if let Some(b) = s.random_baseline {
                parts.push(format!(
                    "random agent over {} voyages: reaches {} {:.0}% · mean return {}",
                    b.episodes,
                    s.goal_name,
                    b.success_rate * 100.0,
                    signed(f64::from(b.mean_return))
                ));
            }
            parts.join(" · ")
        }
    }
}

/// The settings a guided experiment pins, in one line.
fn experiment_settings(def: &rl_tutorial::ExperimentDef) -> String {
    let d = QLearningConfig::default();
    let mut parts = vec![format!(
        "α {} · γ {} · ε {}",
        def.alpha.unwrap_or(d.alpha),
        def.gamma.unwrap_or(d.gamma),
        def.epsilon.unwrap_or(d.epsilon)
    )];
    parts.extend(def.rewards.iter().map(|(k, v)| format!("{k} {}", signed(f64::from(*v)))));
    if let Some(features) = &def.observation {
        parts.push(format!("observes {}", features.iter().map(|f| f.name()).collect::<Vec<_>>().join(" + ")));
    }
    parts.push(format!("{} episodes · training seed {}", def.episodes, def.training_seed));
    parts.join(" · ")
}

fn experiment_summary(r: &ExperimentResult, goal: &str) -> String {
    let mut s = format!("{} of {} test voyages reached {goal}", r.reached, r.tested);
    if let Some(shortest) = r.shortest {
        s.push_str(&format!(" · {shortest} took the shortest route"));
    }
    s.push_str(&format!(" · {:.1} moves on average", r.mean_length));
    s
}

const COMPASS: [(&str, &str); 4] = [("N", "north"), ("E", "east"), ("S", "south"), ("W", "west")];

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
    let just_completed = RwSignal::new(false);
    let report = RwSignal::new(None::<BatchReport>);
    let batch_running = RwSignal::new(false);
    let batch_size = RwSignal::new(100u32);
    let batch_first_seed = RwSignal::new(1u64);
    let feedback = RwSignal::new(std::collections::BTreeMap::<(usize, usize), AnswerOutcome>::new());
    let error = RwSignal::new(None::<String>);
    let seed = RwSignal::new(params.seed.unwrap_or(1));
    let view = RwSignal::new(params.view);
    let plain = RwSignal::new(params.plain);
    let depth = RwSignal::new(Depth::Intuition);
    let tutorial_index = RwSignal::new(0usize);
    let pending_moves = StoredValue::new(Some(params.moves));
    let pending_step = StoredValue::new(params.tutorial_step);
    let pending_watch = StoredValue::new(params.watch);
    let pending_batch = StoredValue::new(params.batch);
    let pending_rewards = StoredValue::new(params.rewards);
    let pending_train = StoredValue::new(params.train);
    let pending_deploy = StoredValue::new(params.deploy);
    let first_load = StoredValue::new(true);

    // ---- training state
    let draft = RwSignal::new(None::<Draft>);
    let sync_draft = StoredValue::new(true);
    let training = RwSignal::new(false);
    let deploy_report = RwSignal::new(None::<DeployReport>);
    let show_policy = RwSignal::new(true);
    let selected_cell = RwSignal::new(None::<(i32, i32)>);

    // ---- walkthrough state
    let lesson_line = RwSignal::new(0usize);
    let lesson_reveal = StoredValue::new(None::<Reveal>);
    let lesson_play = RwSignal::new(None::<Play>);
    let updated_mark = StoredValue::new(None::<Updated>);
    let flashed = StoredValue::new(None::<(u32, u32)>);

    // ---- tutorial predictions and guided experiments, by step index
    let predictions = RwSignal::new(BTreeMap::<usize, usize>::new());
    let experiments = RwSignal::new(BTreeMap::<usize, ExperimentResult>::new());
    let experiment_running = RwSignal::new(None::<usize>);
    let guided_label = StoredValue::new(None::<String>);
    let guided_started = StoredValue::new(0.0f64);

    // ---- experiment log and comparison
    let experiment_log = RwSignal::new(ExperimentLog::new(&level_id.get_untracked()));
    let log_loaded = StoredValue::new(None::<String>);
    let selection = RwSignal::new(Selection::default());
    let compare_pending = RwSignal::new(None::<Pending>);
    let reruns = RwSignal::new(BTreeMap::<u32, bool>::new());
    let compare_episodes = RwSignal::new(500u32);

    // ---- saved rafts
    let saved_key = StoredValue::new(None::<String>);

    // ---- simulation worker
    // The worker's callback must send follow-up requests (the next training
    // chunk), but the bridge does not exist yet when the callback is built:
    // it reaches the bridge through this slot, filled right after spawning.
    let bridge_slot = StoredValue::new_local(None::<gloo_worker::WorkerBridge<SimWorker>>);
    let bridge_send = move |request: SimRequest| {
        bridge_slot.with_value(|b| {
            if let Some(b) = b {
                b.send(request)
            }
        })
    };
    // Training runs on a target: this browser's worker for now.
    let target =
        StoredValue::new_local(Rc::new(LocalWorker::new(TRAIN_CHUNK, Rc::new(bridge_send))) as Rc<dyn TrainingTarget>);
    let target_label = target.with_value(|t| t.label());
    let bridge = SimWorker::spawner()
        .callback(move |response| match response {
            SimResponse::Snapshot(s) => {
                error.set(None);
                snapshot.set(Some(*s));
            }
            SimResponse::Trained(s) => {
                error.set(None);
                snapshot.set(Some(*s));
                let more = training.get_untracked() && !target.with_value(|t| t.chunk_done());
                if !more {
                    training.set(false);
                    if pending_deploy.try_update_value(std::mem::take).unwrap_or(false) {
                        bridge_send(SimRequest::Deploy);
                    }
                }
            }
            SimResponse::Lesson(s) => {
                error.set(None);
                let s = *s;
                if let (Some(reveal), Some(l)) = (lesson_reveal.try_update_value(Option::take).flatten(), &s.lesson) {
                    lesson_line.set(lesson_code::line_after(reveal, l));
                }
                let finished = s.lesson.as_ref().is_some_and(|l| l.finished.is_some());
                snapshot.set(Some(s));
                if let Some(play) = lesson_play.get_untracked() {
                    if play == Play::Watch && finished {
                        lesson_play.set(None);
                    } else {
                        set_timeout(
                            move || {
                                if lesson_play.get_untracked() == Some(play)
                                    && lesson_reveal.with_value(Option::is_none)
                                {
                                    let (mode, reveal) = play.request();
                                    lesson_reveal.set_value(Some(reveal));
                                    bridge_send(SimRequest::Lesson { mode });
                                }
                            },
                            Duration::from_millis(play.delay_ms()),
                        );
                    }
                }
            }
            SimResponse::Experiment { result, snapshot: s, run } => {
                error.set(None);
                let pending = compare_pending.get_untracked();
                compare_pending.set(None);
                let adopted = |s: Box<Snapshot>| {
                    sync_draft.set_value(true);
                    deploy_report.set(None);
                    snapshot.set(Some(*s));
                    show_policy.set(true);
                    view.set(View::Arena);
                };
                if let Some(step) = experiment_running.get_untracked() {
                    experiment_running.set(None);
                    experiments.update(|e| {
                        e.insert(step, result);
                    });
                    let label = guided_label.try_update_value(Option::take).flatten();
                    let ms = now_ms() - guided_started.get_value();
                    experiment_log.update(|l| {
                        l.add(&run, label, Some(ms));
                    });
                    adopted(s);
                    return;
                }
                match pending {
                    Some(Pending::New { started_ms }) => {
                        let id = experiment_log.try_update(|l| l.add(&run, None, Some(now_ms() - started_ms)));
                        if let Some(id) = id {
                            selection.update(|sel| {
                                sel.toggle(id);
                            });
                        }
                        snapshot.set(Some(*s));
                    }
                    Some(Pending::Rerun { id }) => {
                        let same = experiment_log
                            .with_untracked(|l| l.get(id).is_some_and(|r| r.fingerprint == run.fingerprint()));
                        reruns.update(|m| {
                            m.insert(id, same);
                        });
                        snapshot.set(Some(*s));
                    }
                    Some(Pending::Adopt { .. }) => adopted(s),
                    None => snapshot.set(Some(*s)),
                }
            }
            SimResponse::Export(save) => {
                if let Ok(json) = serde_json::to_string(&*save) {
                    leptos::task::spawn_local(async move { store::save(&save.level_id, &json).await });
                }
            }
            SimResponse::Deploy(r) => {
                let r = *r;
                let newly = snapshot.with_untracked(|s| {
                    let s = s.as_ref()?;
                    let goal = &s.objective.as_ref()?.goal;
                    let mut newly = false;
                    progress.update(|p| newly = p.record_deploy(&s.level_id, goal, &r));
                    Some(newly)
                });
                if newly == Some(true) {
                    just_completed.set(true);
                }
                deploy_report.set(Some(r));
            }
            SimResponse::Batch(r) => {
                batch_running.set(false);
                let r = *r;
                let newly = snapshot.with_untracked(|s| {
                    let s = s.as_ref()?;
                    let goal = &s.objective.as_ref()?.goal;
                    let mut newly = false;
                    progress.update(|p| newly = p.record_batch(&s.level_id, goal, &r));
                    Some(newly)
                });
                if newly == Some(true) {
                    just_completed.set(true);
                }
                report.set(Some(r));
            }
            SimResponse::Error(e) => {
                batch_running.set(false);
                training.set(false);
                target.with_value(|t| t.cancel());
                lesson_play.set(None);
                lesson_reveal.set_value(None);
                experiment_running.set(None);
                compare_pending.set(None);
                error.set(Some(e));
            }
        })
        // Trunk builds the worker as a classic (non-module) script.
        .as_module(false)
        .spawn("./worker.js");
    bridge_slot.set_value(Some(bridge));
    let send = bridge_send;

    // ---- autopilot: an agent sails while the player watches
    let autopilot = StoredValue::new_local(None::<IntervalHandle>);
    let autopilot_on = RwSignal::new(false);
    let autopilot_trained = RwSignal::new(false);
    let autopilot_replay = RwSignal::new(false);
    let stop_autopilot = move || {
        autopilot.update_value(|h| {
            if let Some(h) = h.take() {
                h.clear();
            }
        });
        autopilot_on.set(false);
    };
    let start_autopilot = move |trained: bool| {
        stop_autopilot();
        send(SimRequest::Reset { seed: seed.get_untracked() });
        autopilot_trained.set(trained);
        autopilot_replay.set(false);
        if let Ok(h) = set_interval_with_handle(
            move || send(SimRequest::AgentStep { trained }),
            Duration::from_millis(AUTOPILOT_MS),
        ) {
            autopilot.set_value(Some(h));
            autopilot_on.set(true);
        }
    };

    // A recorded voyage, played back move by move from its action log.
    let start_replay = move |voyage_seed: u64, actions: Vec<u32>, edits: PlayerEdits| {
        stop_autopilot();
        seed.set(voyage_seed);
        send(SimRequest::Replay { seed: voyage_seed, actions, edits });
        autopilot_trained.set(false);
        autopilot_replay.set(true);
        view.set(View::Arena);
        if let Ok(h) = set_interval_with_handle(
            move || send(SimRequest::AgentStep { trained: false }),
            Duration::from_millis(AUTOPILOT_MS),
        ) {
            autopilot.set_value(Some(h));
            autopilot_on.set(true);
        }
    };

    // ---- tutorial
    let tutorial =
        Memo::new(move |_| content::level(&level_id.get()).and_then(|e| TutorialDef::parse(e.tutorial_source).ok()));
    let highlights = Memo::new(move |_| {
        let i = tutorial_index.get();
        tutorial.with(|t| t.as_ref().and_then(|t| t.steps.get(i)).map(|s| s.highlight().to_vec()).unwrap_or_default())
    });
    let is_hl = move |el: UiElement| highlights.with(|h| h.contains(&el));
    let step_count = move || tutorial.with(|t| t.as_ref().map_or(0, |t| t.steps.len()));
    let current_step = move || {
        let i = tutorial_index.get();
        let d = depth.get();
        tutorial.with(|t| {
            t.as_ref()
                .and_then(|t| t.steps.get(i))
                .map(|s| (s.beat().label().to_string(), s.title().to_string(), s.text().at(d).to_string()))
        })
    };

    // ---- walkthrough: Q-learning one line, one move or one episode at a time
    // One request at a time: input while an answer is pending is ignored, so
    // fast key presses cannot skip ahead of what the panel shows.
    let lesson_request = move |mode: LessonMode, reveal: Reveal| {
        if lesson_reveal.with_value(Option::is_some) {
            return;
        }
        lesson_reveal.set_value(Some(reveal));
        send(SimRequest::Lesson { mode });
    };
    let current_lesson = move || snapshot.with_untracked(|s| s.as_ref().and_then(|s| s.lesson.clone()));
    let step_line = move || {
        lesson_play.set(None);
        if lesson_reveal.with_value(Option::is_some) {
            return;
        }
        match lesson_code::step_line(current_lesson().as_ref(), lesson_line.get_untracked()) {
            LineStep::Show(line) => lesson_line.set(line),
            LineStep::Request(mode, reveal) => lesson_request(mode, reveal),
        }
    };
    let step_move = move || {
        lesson_play.set(None);
        lesson_request(LessonMode::Move, Reveal::Move);
    };
    let open_walkthrough = move || {
        stop_autopilot();
        view.set(View::Walkthrough);
        if current_lesson().is_none() {
            lesson_request(LessonMode::Start, Reveal::Start);
        }
    };
    let walk_ctx = WalkCtx {
        snapshot,
        depth,
        line: lesson_line,
        play: lesson_play,
        highlights,
        step_line: Callback::new(move |()| step_line()),
        step_move: Callback::new(move |()| step_move()),
        finish_episode: Callback::new(move |()| {
            lesson_play.set(None);
            lesson_request(LessonMode::Episode, Reveal::Episode);
        }),
        set_play: Callback::new(move |p: Option<Play>| {
            lesson_play.set(p);
            if let Some(p) = p {
                let (mode, reveal) = p.request();
                lesson_request(mode, reveal);
            }
        }),
        back: Callback::new(move |()| {
            lesson_play.set(None);
            view.set(View::Train);
        }),
    };
    Effect::new(move |_| {
        if view.get() != View::Walkthrough {
            lesson_play.set(None);
        }
    });
    // The Q-value just updated flashes on the map once its line is reached.
    Effect::new(move |_| {
        let line = lesson_line.get();
        snapshot.with(|s| {
            let Some(l) = s.as_ref().and_then(|s| s.lesson.as_ref()) else { return };
            let Some(m) = &l.last else { return };
            let key = (l.episode, l.moves);
            if line >= UPDATE && flashed.get_value() != Some(key) {
                flashed.set_value(Some(key));
                updated_mark.set_value(Some(Updated { cell: m.from, action: m.trace.action, born: now_ms() }));
            }
        });
    });
    // What the arena shows during the walkthrough, given the current line.
    let lesson_model = Memo::new(move |_| {
        let line = lesson_line.get();
        snapshot.with(|s| {
            s.as_ref()
                .and_then(|s| s.lesson.as_ref())
                .map(|l| (lesson_code::displayed_model(l, line), l.last.is_none()))
        })
    });
    let lesson_render_state = StoredValue::new_local(RenderState::default());
    Effect::new(move |_| {
        lesson_model.with(|m| {
            if let Some((model, fresh)) = m {
                lesson_render_state.update_value(|rs| rs.on_snapshot(model, now_ms(), *fresh));
            }
        })
    });
    // The Q-table as shown: during the walkthrough, as of the current line.
    let shown_policy = Memo::new(move |_| {
        let v = view.get();
        let line = lesson_line.get();
        snapshot.with(|s| {
            let s = s.as_ref()?;
            let t = s.training.as_ref()?;
            if t.policy.is_empty() || s.replaying {
                return None;
            }
            Some(match (v, &s.lesson) {
                (View::Walkthrough, Some(l)) => lesson_code::displayed_policy(&t.policy, l, line),
                _ => t.policy.clone(),
            })
        })
    });
    // The cell whose Q-values the inspector shows: the selected one, or the raft's.
    let inspected = Memo::new(move |_| {
        let selected = selected_cell.get();
        let cell = selected.or_else(|| {
            snapshot.with(|s| {
                let s = s.as_ref()?;
                match (view.get(), &s.lesson) {
                    (View::Walkthrough, Some(l)) => Some(
                        l.last.as_ref().filter(|_| lesson_line.get() < lesson_code::ADVANCE).map_or(l.cell, |m| m.from),
                    ),
                    _ => Some(s.position),
                }
            })
        })?;
        shown_policy.with(|p| p.as_ref()?.iter().find(|c| c.cell == cell).copied()).map(|c| (c, selected.is_some()))
    });

    // ---- level loading
    Effect::new(move |_| {
        let id = level_id.get();
        stop_autopilot();
        just_completed.set(false);
        report.set(None);
        feedback.set(Default::default());
        training.set(false);
        target.with_value(|t| t.cancel());
        deploy_report.set(None);
        sync_draft.set_value(true);
        selected_cell.set(None);
        lesson_play.set(None);
        lesson_line.set(0);
        predictions.set(Default::default());
        experiments.set(Default::default());
        experiment_running.set(None);
        saved_key.set_value(None);
        // This level's experiment log.
        experiment_log.set(ExperimentLog::new(&id));
        log_loaded.set_value(None);
        selection.set(Selection::default());
        reruns.set(Default::default());
        compare_pending.set(None);
        compare_episodes.set(
            content::level(&id)
                .and_then(|e| rl_envs::LevelDef::parse(e.level_source).ok())
                .and_then(|d| d.budget.map(|b| b.episodes))
                .unwrap_or(500),
        );
        {
            let id = id.clone();
            leptos::task::spawn_local(async move {
                let saved = store::load(&ExperimentLog::storage_key(&id)).await;
                if level_id.get_untracked() != id {
                    return;
                }
                log_loaded.set_value(Some(id.clone()));
                if let Some(log) = saved.and_then(|json| ExperimentLog::from_json(&id, &json)) {
                    experiment_log.set(log);
                }
            });
        }
        // The first load keeps the view from the URL; later level changes return to the arena.
        if !first_load.try_update_value(std::mem::take).unwrap_or(false) {
            view.update(|v| {
                if v.is_panel() {
                    *v = View::Arena;
                }
            });
        }
        tutorial_index.set(pending_step.try_update_value(|s| s.take()).flatten().unwrap_or(0));
        send(SimRequest::Load { level_id: id.clone(), seed: seed.get_untracked() });
        let scripted = pending_rewards.with_value(Option::is_some) || pending_train.with_value(Option::is_some);
        if !scripted {
            // Bring back the raft saved for this level, if any.
            leptos::task::spawn_local(async move {
                let Some(json) = store::load(&id).await else { return };
                let Ok(save) = serde_json::from_str::<SavedRaft>(&json) else { return };
                if level_id.get_untracked() == id {
                    sync_draft.set_value(true);
                    send(SimRequest::Restore(Box::new(save)));
                }
            });
        }
        if view.get_untracked() == View::Walkthrough {
            lesson_reveal.set_value(Some(Reveal::Start));
            send(SimRequest::Lesson { mode: LessonMode::Start });
        }
        for action in pending_moves.try_update_value(|m| m.take()).flatten().unwrap_or_default() {
            send(SimRequest::Step { action });
        }
        if pending_watch.try_update_value(std::mem::take).unwrap_or(false) {
            start_autopilot(false);
        }
        if let Some(rewards) = pending_rewards.try_update_value(Option::take).flatten() {
            send(SimRequest::Configure {
                config: QLearningConfig::default(),
                rewards,
                observation: None,
                training_seed: 1,
            });
        }
        if let Some(episodes) = pending_train.try_update_value(Option::take).flatten() {
            training.set(true);
            target.with_value(|t| t.start(episodes));
        }
        if let Some(episodes) = pending_batch.try_update_value(Option::take).flatten() {
            batch_size.set(episodes);
            batch_running.set(true);
            send(SimRequest::Batch { first_seed: batch_first_seed.get_untracked(), episodes });
        }
    });

    // ---- campaign progress
    Effect::new(move |_| progress.with(save_progress));
    // A trained raft is saved after every finished training run or walkthrough episode.
    Effect::new(move |_| {
        let busy = training.get();
        snapshot.with(|s| {
            let Some(s) = s.as_ref() else { return };
            let Some(t) = s.training.as_ref() else { return };
            if busy || t.episodes_trained == 0 {
                return;
            }
            let key =
                format!("{}|{}|{}|{:?}|{:?}", s.level_id, t.episodes_trained, t.training_seed, t.config, t.rewards);
            if saved_key.get_value().as_deref() != Some(key.as_str()) {
                saved_key.set_value(Some(key));
                send(SimRequest::Export);
            }
        })
    });
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
        match s.sailed_by {
            SailedBy::RandomAgent => report.with_untracked(|r| {
                if let Some(r) = r {
                    progress.update(|p| newly_completed = p.record_replay(&s, r));
                }
            }),
            _ => progress.update(|p| newly_completed = p.record(&s)),
        }
        if newly_completed {
            just_completed.set(true);
        }
    });
    // A completed level opens its debrief.
    Effect::new(move |_| {
        if just_completed.get() {
            let debrief = tutorial
                .with_untracked(|t| t.as_ref().and_then(|t| t.steps.iter().position(|s| s.beat() == Beat::Debrief)));
            if let Some(i) = debrief {
                tutorial_index.set(i);
            }
        }
    });

    // ---- analyse: batch experiments and questions
    let analyse_ctx = AnalyseCtx {
        report,
        running: batch_running,
        batch_size,
        first_seed: batch_first_seed,
        snapshot,
        progress,
        feedback,
        highlights,
        run_batch: Callback::new(move |()| {
            batch_running.set(true);
            send(SimRequest::Batch {
                first_seed: batch_first_seed.get_untracked(),
                episodes: batch_size.get_untracked(),
            });
        }),
        replay_best: Callback::new(move |best_seed: u64| {
            seed.set(best_seed);
            view.set(View::Arena);
            start_autopilot(false);
        }),
        submit_answer: Callback::new(move |(task, question, answer): (usize, usize, Answer)| {
            let Some((level, goal)) = snapshot.with_untracked(|s| {
                s.as_ref().and_then(|s| Some((s.level_id.clone(), s.objective.as_ref()?.goal.clone())))
            }) else {
                return;
            };
            let mut result = (AnswerOutcome::Incorrect, false);
            report.with_untracked(|r| {
                progress.update(|p| result = p.answer(&level, &goal, (task, question), answer, r.as_ref()));
            });
            feedback.update(|f| {
                f.insert((task, question), result.0);
            });
            if result.1 {
                just_completed.set(true);
            }
        }),
    };
    // ---- training: settings draft, training runs, deployment
    Effect::new(move |_| {
        snapshot.with(|s| {
            if let Some(t) = s.as_ref().and_then(|s| s.training.as_ref())
                && sync_draft.get_value()
            {
                draft.set(Some(Draft::from_status(t)));
                sync_draft.set_value(false);
            }
        })
    });
    // A new experiment from the settings in CONFIGURE; the current raft is untouched.
    let run_new_experiment = move || {
        if compare_pending.with_untracked(Option::is_some) {
            return;
        }
        let id = level_id.get_untracked();
        let config = draft.with_untracked(|d| {
            snapshot.with_untracked(|s| {
                let status = s.as_ref()?.training.as_ref()?;
                Some(d.as_ref()?.experiment(&id, status, compare_episodes.get_untracked()))
            })
        });
        match config {
            Some(Ok(config)) => {
                compare_pending.set(Some(Pending::New { started_ms: now_ms() }));
                send(SimRequest::Experiment { config: Box::new(config), adopt: false });
            }
            Some(Err(e)) => error.set(Some(e)),
            None => {}
        }
    };
    // A comparison task is settled by the log: check it whenever the log changes.
    Effect::new(move |_| {
        experiment_log.with(|log| {
            let Some(def) = content::level(&log.level_id).and_then(|e| rl_envs::LevelDef::parse(e.level_source).ok())
            else {
                return;
            };
            let Some(goal) = def.challenge.map(|c| c.goal) else { return };
            let mut newly = false;
            progress.update(|p| newly = p.record_experiments(&goal, log, &def.observation));
            if newly {
                just_completed.set(true);
            }
        })
    });
    // The log is saved whenever it changes, once this level's saved log has been read.
    Effect::new(move |_| {
        experiment_log.with(|log| {
            if log_loaded.get_value().as_deref() != Some(log.level_id.as_str()) {
                return;
            }
            if let Ok(json) = serde_json::to_string(log) {
                let key = ExperimentLog::storage_key(&log.level_id);
                leptos::task::spawn_local(async move { store::save(&key, &json).await });
            }
        })
    });
    let train_ctx = TrainCtx {
        snapshot,
        progress,
        highlights,
        draft,
        training,
        deploy: deploy_report,
        apply: Callback::new(move |()| {
            let Some(d) = draft.get_untracked() else { return };
            stop_autopilot();
            training.set(false);
            target.with_value(|t| t.cancel());
            lesson_play.set(None);
            let id = level_id.get_untracked();
            leptos::task::spawn_local(async move { store::forget(&id).await });
            deploy_report.set(None);
            sync_draft.set_value(true);
            send(SimRequest::Configure {
                config: d.config,
                rewards: d.rewards,
                observation: d.observation,
                training_seed: d.training_seed,
            });
        }),
        train: Callback::new(move |episodes: u32| {
            training.set(true);
            target.with_value(|t| t.start(episodes));
        }),
        stop_training: Callback::new(move |()| {
            target.with_value(|t| t.cancel());
            training.set(false);
        }),
        open_walkthrough: Callback::new(move |()| open_walkthrough()),
        deploy_now: Callback::new(move |()| send(SimRequest::Deploy)),
        watch_trained: Callback::new(move |first_seed: u64| {
            seed.set(first_seed);
            view.set(View::Arena);
            start_autopilot(true);
        }),
        run_experiment: Callback::new(move |()| {
            run_new_experiment();
            view.set(View::Compare);
        }),
    };
    let compare_ctx = CompareCtx {
        snapshot,
        log: experiment_log,
        selection,
        pending: compare_pending,
        reruns,
        draft,
        episodes: compare_episodes,
        highlights,
        run: Callback::new(move |()| run_new_experiment()),
        rerun: Callback::new(move |id: u32| {
            let Some(config) = experiment_log.with_untracked(|l| l.get(id).map(|r| r.config.clone())) else { return };
            reruns.update(|m| {
                m.remove(&id);
            });
            compare_pending.set(Some(Pending::Rerun { id }));
            send(SimRequest::Experiment { config: Box::new(config), adopt: false });
        }),
        adopt: Callback::new(move |id: u32| {
            let Some(config) = experiment_log.with_untracked(|l| l.get(id).map(|r| r.config.clone())) else { return };
            stop_autopilot();
            training.set(false);
            target.with_value(|t| t.cancel());
            lesson_play.set(None);
            compare_pending.set(Some(Pending::Adopt { id }));
            send(SimRequest::Experiment { config: Box::new(config), adopt: true });
        }),
        remove: Callback::new(move |id: u32| {
            experiment_log.update(|l| l.remove(id));
            experiment_log.with_untracked(|l| selection.update(|s| s.retain_in(l)));
            reruns.update(|m| {
                m.remove(&id);
            });
        }),
        watch: Callback::new(move |(id, k): (u32, usize)| {
            let voyage = experiment_log.with_untracked(|l| {
                let r = l.get(id)?;
                let e = r.evaluation.episodes.get(k)?;
                Some((e.played.seed, e.actions.clone(), r.config.edits.clone()))
            });
            if let Some((voyage_seed, actions, edits)) = voyage {
                start_replay(voyage_seed, actions, edits);
            }
        }),
        edit_settings: Callback::new(move |()| view.set(View::Configure)),
    };
    let trained =
        move || snapshot.with(|s| s.as_ref().and_then(|s| s.training.as_ref()).is_some_and(|t| t.episodes_trained > 0));

    let task_states = move || {
        snapshot.with(|s| {
            s.as_ref()
                .and_then(|s| {
                    let goal = &s.objective.as_ref()?.goal;
                    Some(progress.with(|p| p.task_states(&s.level_id, goal)))
                })
                .unwrap_or_default()
        })
    };
    let level_complete = move || progress.with(|p| p.is_complete(&level_id.get()));
    let next_level = move || level_index(&level_id.get()).map(|i| i + 1).filter(|&i| i < LEVELS.len());
    let pilot = move || snapshot.with(|s| s.as_ref().map(|s| s.pilot));

    // ---- phases appear as the tutorial reaches them
    // The furthest tutorial step reached in this level: going back keeps the phases already shown.
    let reached = RwSignal::new((String::new(), 0usize));
    Effect::new(move |_| {
        let (id, i) = (level_id.get_untracked(), tutorial_index.get());
        reached.update(|r| *r = if r.0 == id { (id, r.1.max(i)) } else { (id, i) });
    });
    // A stage of phases is shown from the first step that points into it. A stage the tutorial
    // never points into is shown from the start.
    let stage_reached = move |stage: Stage| {
        let first = tutorial.with(|t| {
            t.as_ref()?.steps.iter().position(|s| s.highlight().iter().any(|&e| Stage::of(e) == Some(stage)))
        });
        first.is_none_or(|i| i <= reached.with(|r| r.1))
    };
    let phase_shown = move |v: View| {
        if pilot() != Some(Pilot::Agent) {
            return false;
        }
        let here = view.get() == v || (v == View::Train && view.get() == View::Walkthrough);
        let stage = Stage::of_view(v);
        here || level_complete()
            || trained()
            || stage_reached(stage)
            || (stage == Stage::Measure && task_states().iter().any(|t| t.done))
    };
    // The trained raft's controls and its Q-values belong to the learning stage too.
    let learning = move || phase_shown(View::Train);

    // ---- decluttering: the objective opens on demand, side panels fold
    let objective_open = RwSignal::new(false);
    Effect::new(move |_| {
        if just_completed.get() || is_hl(UiElement::Objective) {
            objective_open.set(true);
        }
    });
    let open_level = move |i: usize| {
        let id = LEVELS[i].id;
        if level_id.get_untracked() == id {
            objective_open.update(|o| *o = !*o);
        } else {
            level_id.set(id.to_string());
            objective_open.set(true);
        }
    };
    // The concept library opens a concept's explainer step in its level.
    let library_ctx = LibraryCtx {
        progress,
        depth,
        unlock_all,
        replay: Callback::new(move |(i, step): (usize, usize)| {
            if !can_open(i) {
                return;
            }
            let id = LEVELS[i].id;
            if level_id.get_untracked() == id {
                tutorial_index.set(step);
            } else {
                pending_step.set_value(Some(step));
                level_id.set(id.to_string());
            }
            view.set(View::Arena);
        }),
    };
    // Reaching a tutorial step unlocks the concepts it teaches.
    Effect::new(move |_| {
        // Tracks the step only: the level loader always sets it, after the level changes.
        let (id, i) = (level_id.get_untracked(), tutorial_index.get());
        if progress.with_untracked(|p| p.reached.get(&id).is_none_or(|&r| i > r)) {
            progress.update(|p| {
                p.reach(&id, i);
            });
        }
    });
    // The player's own fold choices, until the view changes. A tutorial
    // highlight inside a panel unfolds it.
    let fold_choice = RwSignal::new([None::<bool>; 3]);
    Effect::new(move |prev: Option<View>| {
        let v = view.get();
        highlights.with(|h| {
            fold_choice.update(|c| {
                if prev != Some(v) {
                    *c = [None; 3];
                }
                for p in SidePanel::ALL {
                    if p.elements().iter().any(|e| h.contains(e)) {
                        c[p as usize] = Some(false);
                    }
                }
            });
        });
        v
    });
    let folded = move |p: SidePanel| {
        Signal::derive(move || fold_choice.with(|c| c[p as usize]).unwrap_or_else(|| p.folded_in(view.get())))
    };
    let toggle_fold = move |p: SidePanel| {
        Callback::new(move |()| {
            let now = folded(p).get_untracked();
            fold_choice.update(|c| c[p as usize] = Some(!now));
        })
    };

    // ---- keyboard
    let _keys = window_event_listener(ev::keydown, move |e: web_sys::KeyboardEvent| {
        let tag = e.target().and_then(|t| t.dyn_into::<web_sys::Element>().ok()).map(|el| el.tag_name());
        if matches!(tag.as_deref(), Some("INPUT" | "SELECT" | "TEXTAREA")) {
            return;
        }
        if e.key() == "Escape" && objective_open.get_untracked() {
            objective_open.set(false);
            return;
        }
        if view.get_untracked() == View::Walkthrough && tag.as_deref() != Some("BUTTON") {
            match e.key().as_str() {
                " " => {
                    e.prevent_default();
                    step_line();
                    return;
                }
                "Enter" => {
                    e.prevent_default();
                    step_move();
                    return;
                }
                _ => {}
            }
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
                send(SimRequest::Reset { seed: seed.get_untracked() });
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
            let fx = Effects { plain: plain.get_untracked() };
            let hl = highlights.with_untracked(|h| Highlights {
                agent: h.contains(&UiElement::Agent),
                goal: h.contains(&UiElement::Goal),
            });
            match view.get_untracked() {
                View::StyleSheet => render::draw_style_sheet(&ctx, now, fx),
                View::Configure | View::Train | View::Analyse | View::Deploy | View::Compare | View::Library => {}
                View::Walkthrough => lesson_model.with_untracked(|m| {
                    if let Some((model, _)) = m {
                        let policy = shown_policy.get_untracked();
                        let marks =
                            Marks { selected: selected_cell.get_untracked(), updated: updated_mark.get_value() };
                        lesson_render_state.update_value(|rs| {
                            render::draw_world(
                                &ctx,
                                model,
                                rs,
                                now,
                                fx,
                                Layers { hl, policy: policy.as_deref(), marks },
                            )
                        });
                    }
                }),
                View::Arena => snapshot.with_untracked(|s| {
                    if let Some(s) = s {
                        let trained = s.training.as_ref().is_some_and(|t| t.episodes_trained > 0);
                        let policy =
                            (trained && show_policy.get_untracked()).then(|| shown_policy.get_untracked()).flatten();
                        let marks = Marks { selected: selected_cell.get_untracked(), updated: None };
                        render_state.update_value(|rs| {
                            render::draw_world(
                                &ctx,
                                &s.render,
                                rs,
                                now,
                                fx,
                                Layers { hl, policy: policy.as_deref(), marks },
                            )
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
    // During the walkthrough the RL loop panel follows the walkthrough episode,
    // as of the highlighted line.
    let hud = move |arena: fn(&Snapshot) -> String, lesson: fn(&LessonState, usize) -> String| {
        move || {
            let walking = view.get() == View::Walkthrough;
            let line = lesson_line.get();
            snapshot.with(|s| {
                let Some(s) = s.as_ref() else { return "—".to_string() };
                match (&s.lesson, walking) {
                    (Some(l), true) => lesson(l, line),
                    _ => arena(s),
                }
            })
        }
    };

    // A tutorial step's prediction and guided experiment, if it has them.
    let step_extra = move |i: usize| {
        let step = tutorial.with_untracked(|t| t.as_ref().and_then(|t| t.steps.get(i).cloned()))?;
        let def = step.experiment()?.clone();
        let settings = experiment_settings(&def);
        let spec = StoredValue::new(experiment_spec(&level_id.get_untracked(), &def));
        let title = step.title().to_string();
        let prediction = match &step {
            StepDef::Predict { question, options, answer, explain, .. } => {
                Some((question.clone(), options.clone(), *answer, explain.clone()))
            }
            _ => None,
        };
        let needs_choice = prediction.is_some();
        let chosen = move || predictions.with(|p| p.get(&i).copied());
        let result = move || experiments.with(|e| e.get(&i).copied());
        let run = move |_| {
            stop_autopilot();
            training.set(false);
            target.with_value(|t| t.cancel());
            lesson_play.set(None);
            match spec.get_value() {
                Ok(config) => {
                    experiment_running.set(Some(i));
                    guided_label.set_value(Some(title.clone()));
                    guided_started.set_value(now_ms());
                    send(SimRequest::Experiment { config: Box::new(config), adopt: true });
                }
                Err(e) => error.set(Some(e)),
            }
        };
        let prediction_view = prediction.map(|(question, options, answer, explain)| {
            let right = StoredValue::new(options[answer].clone());
            let explain = StoredValue::new(explain);
            view! {
                <p class="predict-question">{question}</p>
                <div class="predict-options">
                    {options
                        .into_iter()
                        .enumerate()
                        .map(|(k, o)| {
                            view! {
                                <button
                                    class="option"
                                    class:chosen=move || chosen() == Some(k)
                                    disabled=move || chosen().is_some()
                                    on:click=move |_| predictions.update(|p| {
                                        p.insert(i, k);
                                    })
                                >
                                    {o}
                                </button>
                            }
                        })
                        .collect_view()}
                </div>
                {move || {
                    result()
                        .zip(chosen())
                        .map(|(_, c)| {
                            view! {
                                <p class="verdict" class:right=c == answer>
                                    {if c == answer {
                                        "✓ Your prediction was right.".to_string()
                                    } else {
                                        format!("✗ Not this time. The answer: {}", right.get_value())
                                    }}
                                </p>
                                <p class="explain">{explain.get_value()}</p>
                            }
                        })
                }}
            }
        });
        Some(view! {
            <div class="experiment">
                {prediction_view}
                <p class="experiment-settings">{settings}</p>
                <div class="row">
                    <button
                        class="primary"
                        disabled=move || {
                            pilot() != Some(Pilot::Agent) || experiment_running.get().is_some()
                                || (needs_choice && chosen().is_none())
                        }
                        title=move || if needs_choice && chosen().is_none() { "Pick a prediction first" } else { "" }
                        on:click=run
                    >
                        {move || {
                            if experiment_running.get() == Some(i) {
                                "RUNNING…"
                            } else if result().is_some() {
                                "RUN AGAIN"
                            } else {
                                "RUN THE EXPERIMENT"
                            }
                        }}
                    </button>
                </div>
                <p class="subtitle">"Starts a new raft with these settings, replacing the current one."</p>
                {move || {
                    let goal = snapshot.with(|s| s.as_ref().map(|s| s.goal_name.clone())).unwrap_or_default();
                    result().map(|r| view! { <p class="result glow">{experiment_summary(&r, &goal)}</p> })
                }}
            </div>
        })
    };

    // The inspector: the four Q-values of one cell, laid out as a compass.
    let q_inspector = move || {
        inspected.get().map(|(c, selected)| {
            let best = c.values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let low = c.values.iter().copied().fold(f32::INFINITY, f32::min);
            let preference = best - low > 1e-6;
            let best_i = c.values.iter().position(|&v| v == best).unwrap_or(0);
            let entry = move |k: usize| {
                let (short, _) = COMPASS[k];
                let is_best = preference && c.values[k] >= best - 1e-6;
                view! {
                    <span class=format!("compass-{}", short.to_lowercase()) class:best=is_best>
                        {format!("{short} {}", lesson_code::num(c.values[k]))}
                    </span>
                }
            };
            let (whose, hint) =
                if selected { ("cell", "click again to follow the raft") } else { ("raft at", "click a cell") };
            view! {
                <p class="subtitle">{format!("{whose} {} · {hint}", lesson_code::cell(c.cell))}</p>
                <div class="compass">
                    {entry(0)}
                    {entry(3)}
                    <span class="compass-c">
                        {if preference { format!("best: {}", COMPASS[best_i].1) } else { "no preference yet".into() }}
                    </span>
                    {entry(1)}
                    {entry(2)}
                </div>
            }
        })
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
                    class:active=move || view.get() == View::Library
                    title="Every RL concept met so far"
                    on:click=move |_| view.set(View::Library)
                >
                    "LIBRARY"
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
                                aria-expanded=move || (level_id.get() == l.id && objective_open.get()).to_string()
                                on:click=move |_| open_level(i)
                            >
                                {move || if done() { "✓ " } else if can_open(i) { "○ " } else { "■ " }}
                                {l.menu_label}
                            </button>
                        }
                    })
                    .collect_view()}
                <span class="route">"━━ ┄ ┄ ┄"</span>
                <span class="port far">"ITHACA"</span>
                <span style="flex: 1"></span>
                <button
                    class="objective-chip"
                    class:done=level_complete
                    class:active=move || objective_open.get()
                    class:hl-box=move || is_hl(UiElement::Objective)
                    aria-expanded=move || objective_open.get().to_string()
                    title="The level's objective and tasks"
                    on:click=move |_| objective_open.update(|o| *o = !*o)
                >
                    {move || {
                        let tasks = task_states();
                        let done = tasks.iter().filter(|t| t.done).count();
                        let mark = if objective_open.get() { "▴" } else { "▾" };
                        if level_complete() {
                            format!("✓ LEVEL COMPLETE {mark}")
                        } else if tasks.is_empty() {
                            format!("OBJECTIVE {mark}")
                        } else {
                            format!("OBJECTIVE {done}/{} {mark}", tasks.len())
                        }
                    }}
                </button>
            </nav>

            <main
                class="main"
                on:click=move |e: web_sys::MouseEvent| {
                    let inside = e
                        .target()
                        .and_then(|t| t.dyn_into::<web_sys::Element>().ok())
                        .and_then(|el| el.closest(".objective").ok().flatten())
                        .is_some();
                    if !inside {
                        objective_open.set(false);
                    }
                }
            >
                <section
                    class="objective"
                    class:hl-box=move || is_hl(UiElement::Objective)
                    style:display=move || if objective_open.get() { "flex" } else { "none" }
                    aria-label="Objective"
                >
                    <div class="objective-head">
                        <span class="objective-label">"OBJECTIVE"</span>
                        <span style="flex: 1"></span>
                        <button class="quiet" title="Close (Esc)" on:click=move |_| objective_open.set(false)>
                            "✕"
                        </button>
                    </div>
                    <span class="objective-text">
                        {move || {
                            snapshot
                                .with(|s| s.as_ref().and_then(|s| s.objective.as_ref().map(|o| o.text.clone())))
                                .unwrap_or_default()
                        }}
                    </span>
                    <span class="objective-stats">
                        {move || {
                            snapshot
                                .with(|s| s.as_ref().map(|s| progress.with(|p| objective_stats(s, p))))
                                .unwrap_or_default()
                        }}
                    </span>
                    <ul class="tasks">
                        {move || {
                            task_states()
                                .into_iter()
                                .map(|t| {
                                    let glyph = if t.done { "✓" } else if t.available { "○" } else { "■" };
                                    let note = if t.available { "" } else { " · not available yet" };
                                    view! {
                                        <li class="task" class:done=t.done class:locked=!t.available>
                                            {format!("{glyph} {}{note}", t.label)}
                                        </li>
                                    }
                                })
                                .collect_view()
                        }}
                    </ul>
                    {move || {
                        level_complete()
                            .then(|| {
                                view! {
                                    <div class="complete-row">
                                        <span class="complete glow" class:fresh=move || just_completed.get()>
                                            "✓ LEVEL COMPLETE"
                                        </span>
                                        {move || {
                                            next_level()
                                                .filter(|&i| can_open(i))
                                                .map(|i| {
                                                    view! {
                                                        <button class="sail-on" on:click=move |_| open_level(i)>
                                                            {format!("SAIL ON TO {} ▶", LEVELS[i].menu_label)}
                                                        </button>
                                                    }
                                                })
                                        }}
                                    </div>
                                }
                            })
                    }}
                </section>
                <div class="stage" class:walking=move || view.get() == View::Walkthrough>
                    <section
                        class="arena-wrap"
                        class:hl-box=move || is_hl(UiElement::Arena)
                        style:display=move || if view.get().is_panel() { "none" } else { "flex" }
                    >
                        <canvas
                            node_ref=canvas_ref
                            width="720"
                            height="720"
                            aria-label="Arena · click a cell to inspect its Q-values"
                            on:click=move |e: web_sys::MouseEvent| {
                                let Some(canvas) = canvas_ref.get_untracked() else { return };
                                let sx = render::CANVAS_SIZE / f64::from(canvas.client_width().max(1));
                                let sy = render::CANVAS_SIZE / f64::from(canvas.client_height().max(1));
                                let (x, y) = (f64::from(e.offset_x()) * sx, f64::from(e.offset_y()) * sy);
                                let model = if view.get_untracked() == View::Walkthrough {
                                    lesson_model.with_untracked(|m| m.as_ref().map(|m| m.0.clone()))
                                } else {
                                    snapshot.with_untracked(|s| s.as_ref().map(|s| s.render.clone()))
                                };
                                if let Some(cell) = model.and_then(|m| render::cell_at(&m, x, y)) {
                                    selected_cell.update(|c| *c = if *c == Some(cell) { None } else { Some(cell) });
                                }
                            }
                        ></canvas>
                    </section>
                    {move || match view.get() {
                        View::Configure => configure_view(train_ctx).into_any(),
                        View::Train => train_view(train_ctx).into_any(),
                        View::Walkthrough => walkthrough_view(walk_ctx).into_any(),
                        View::Analyse => analyse_view(analyse_ctx).into_any(),
                        View::Deploy => deploy_view(train_ctx).into_any(),
                        View::Compare => compare_view(compare_ctx).into_any(),
                        View::Library => library_view(library_ctx).into_any(),
                        View::Arena | View::StyleSheet => ().into_any(),
                    }}
                </div>

                <aside class="side">
                    <section class="panel">
                        <div class="briefing-head">
                            <span class="beat">{move || current_step().map(|s| s.0)}</span>
                            <span class="depth" role="group" aria-label="Explanation depth">
                                {[(Depth::Story, "STORY"), (Depth::Intuition, "INTUITION"), (Depth::Math, "MATH")]
                                    .into_iter()
                                    .map(|(d, label)| {
                                        view! {
                                            <button
                                                class:active=move || depth.get() == d
                                                aria-pressed=move || (depth.get() == d).to_string()
                                                on:click=move |_| depth.set(d)
                                            >
                                                {label}
                                            </button>
                                        }
                                    })
                                    .collect_view()}
                            </span>
                        </div>
                        <h2 class="glow">{move || current_step().map(|s| s.1)}</h2>
                        <div class="tutorial-text">{move || current_step().map(|s| s.2)}</div>
                        {move || step_extra(tutorial_index.get())}
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

                    <Fold
                        title="THE RL LOOP"
                        folded=folded(SidePanel::Loop)
                        on_toggle=toggle_fold(SidePanel::Loop)
                        summary=Signal::derive(hud(
                            |s| format!("episode {} · step {} · return {}", s.episode, s.step, signed(f64::from(s.episode_return))),
                            |l, line| {
                                let (moves, ret) = lesson_code::displayed_progress(l, line);
                                format!("episode {} · step {moves} · return {}", l.episode, signed(f64::from(ret)))
                            },
                        ))
                        hl=move || false
                    >
                        <div class="hud">
                            <span class="label" class:hl=move || is_hl(UiElement::State)>"STATE"</span>
                            <span class="value">
                                {hud(
                                    |s| match (s.state_index, s.state_count) {
                                        (Some(i), Some(n)) => format!("state {i} of {n} · {}", s.state_text),
                                        _ => s.state_text.clone(),
                                    },
                                    |l, line| {
                                        let (i, c) = lesson_code::displayed_state(l, line);
                                        format!("state {i} · raft {}", lesson_code::cell(c))
                                    },
                                )}
                            </span>
                            <span class="label" class:hl=move || is_hl(UiElement::Action)>"ACTION"</span>
                            <span class="value">
                                {hud(
                                    |s| s.last_action.clone().unwrap_or("—".into()),
                                    |l, line| {
                                        l.last
                                            .as_ref()
                                            .filter(|_| line >= lesson_code::CHOOSE)
                                            .map_or("—".into(), |m| m.action_label.clone())
                                    },
                                )}
                            </span>
                            <span
                                class="label"
                                class:hl=move || is_hl(UiElement::Reward)
                                title="What the sea pays for one move, the moment it is made"
                            >
                                "REWARD"
                            </span>
                            <span class="value">
                                {hud(
                                    |s| reward_text(s.last_reward),
                                    |l, line| {
                                        reward_text(
                                            l.last.as_ref().filter(|_| line >= lesson_code::ACT).map(|m| m.trace.reward),
                                        )
                                    },
                                )}
                            </span>
                            <span
                                class="label"
                                class:hl=move || is_hl(UiElement::Return)
                                title="Every reward of this episode added up: the score a voyage is judged by, and what an agent learns to maximise"
                            >
                                "RETURN"
                            </span>
                            <span class="value">
                                {hud(
                                    |s| return_text(s.episode_return, s.step),
                                    |l, line| {
                                        let (moves, g) = lesson_code::displayed_progress(l, line);
                                        return_text(g, moves)
                                    },
                                )}
                            </span>
                            <span class="label" class:hl=move || is_hl(UiElement::Step)>"STEP"</span>
                            <span class="value">
                                {hud(
                                    |s| format!("{} / {}", s.step, s.max_steps),
                                    |l, line| format!("{} / {}", lesson_code::displayed_progress(l, line).0, l.max_steps),
                                )}
                            </span>
                            <span class="label" class:hl=move || is_hl(UiElement::Episode)>"EPISODE"</span>
                            <span class="value">
                                {hud(|s| {
                                    let arrived = format!("reached {}", s.goal_name);
                                    let status = match s.outcome {
                                        Outcome::Sailing => "sailing",
                                        Outcome::ReachedShore => arrived.as_str(),
                                        Outcome::Died => "swallowed by a whirlpool",
                                        Outcome::Caught => "caught",
                                        Outcome::OutOfTime => "out of time",
                                    };
                                    let pilot = match s.sailed_by {
                                        SailedBy::Nobody => "",
                                        SailedBy::Player => " · by you",
                                        SailedBy::RandomAgent => " · by the random agent",
                                        SailedBy::TrainedAgent => " · by the trained raft",
                                        SailedBy::Replay => " · a recorded voyage",
                                    };
                                    format!("{} · {status}{pilot}", s.episode)
                                },
                                |l, _| format!("training episode {} · walkthrough · seed {}", l.episode, l.seed))}
                            </span>
                        </div>
                    </Fold>

                    {move || {
                        (pilot() == Some(Pilot::Agent) && learning())
                            .then(|| {
                                view! {
                                    <Fold
                                        title="Q-VALUES"
                                        folded=folded(SidePanel::QValues)
                                        on_toggle=toggle_fold(SidePanel::QValues)
                                        summary=Signal::derive(move || {
                                            inspected
                                                .get()
                                                .map(|(c, _)| {
                                                    let best = c.values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                                                    let low = c.values.iter().copied().fold(f32::INFINITY, f32::min);
                                                    let i = c.values.iter().position(|&v| v == best).unwrap_or(0);
                                                    let pref = if best - low > 1e-6 { COMPASS[i].1 } else { "no preference" };
                                                    format!("{} · {pref}", lesson_code::cell(c.cell))
                                                })
                                                .unwrap_or_default()
                                        })
                                        hl=move || is_hl(UiElement::QValues)
                                    >
                                        {move || q_inspector().map_or_else(
                                            || view! { <p class="subtitle">"No Q-table to show yet."</p> }.into_any(),
                                            IntoAny::into_any,
                                        )}
                                    </Fold>
                                }
                            })
                    }}

                    <Fold
                        title="VOYAGE"
                        folded=folded(SidePanel::Voyage)
                        on_toggle=toggle_fold(SidePanel::Voyage)
                        summary=Signal::derive(move || {
                            snapshot
                                .with(|s| s.as_ref().and_then(|s| s.log.last().cloned()))
                                .unwrap_or_else(|| format!("seed {}", seed.get()))
                        })
                        hl=move || false
                    >
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
                                        {move || {
                                            if autopilot_on.get() {
                                                view! { <button class="primary" on:click=move |_| stop_autopilot()>"■ STOP"</button> }
                                                    .into_any()
                                            } else {
                                                view! {
                                                    <button class="primary" on:click=move |_| start_autopilot(false)>
                                                        "▶ WATCH RANDOM AGENT"
                                                    </button>
                                                    <button
                                                        class="primary"
                                                        class:hidden=move || !learning()
                                                        disabled=move || !trained()
                                                        title=move || if trained() { "" } else { "Train a raft first" }
                                                        on:click=move |_| start_autopilot(true)
                                                    >
                                                        "▶ WATCH TRAINED RAFT"
                                                    </button>
                                                }
                                                    .into_any()
                                            }
                                        }}
                                    </div>
                                    <p class="row subtitle">
                                        {move || {
                                            if autopilot_on.get() {
                                                if autopilot_replay.get() {
                                                    "a recorded voyage plays back"
                                                } else if autopilot_trained.get() {
                                                    "the trained raft sails"
                                                } else {
                                                    "the random agent sails"
                                                }
                                            } else {
                                                "steering is off · an agent sails"
                                            }
                                        }}
                                        <label class="toggle" class:hidden=move || !learning()>
                                            <input
                                                type="checkbox"
                                                prop:checked=move || show_policy.get()
                                                on:change=move |_| show_policy.update(|v| *v = !*v)
                                            />
                                            "show learned policy"
                                        </label>
                                    </p>
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
                    </Fold>
                    {move || error.get().map(|e| view! { <p class="panel error">{"ERROR: "}{e}</p> })}
                </aside>
            </main>

            <footer class="phasebar">
                {[
                    (View::Configure, "CONFIGURE", UiElement::Configure, "Learning settings and rewards for a new raft"),
                    (View::Train, "TRAIN", UiElement::Train, "Train the raft and watch its learning curve"),
                    (View::Analyse, "ANALYSE", UiElement::Analyse, "Run batch experiments on the random agent"),
                    (View::Deploy, "DEPLOY", UiElement::Deploy, "Test the trained raft on fixed seeds"),
                    (View::Compare, "COMPARE", UiElement::Compare, "Run experiments and compare them side by side"),
                ]
                    .into_iter()
                    .map(|(v, label, el, about)| {
                        view! {
                            <button
                                class="phase"
                                class:hidden=move || !phase_shown(v)
                                class:active=move || {
                                    view.get() == v || (v == View::Train && view.get() == View::Walkthrough)
                                }
                                class:hl-box=move || is_hl(el)
                                title=about
                                on:click=move |_| view.set(v)
                            >
                                {label}
                            </button>
                        }
                    })
                    .collect_view()}
                <span style="flex: 1"></span>
                <span class="status" class:hidden=move || !phase_shown(View::Train)>
                    {format!("TRAINING ON: {target_label}")}
                </span>
                <button
                    class="quiet"
                    class:active=move || view.get() == View::StyleSheet
                    on:click=move |_| view.set(View::StyleSheet)
                >
                    "STYLE SHEET"
                </button>
                <button
                    class="quiet"
                    on:click=move |_| {
                        if window().confirm_with_message("Reset all campaign progress?").unwrap_or(false) {
                            progress.set(Progress::default());
                            leptos::task::spawn_local(store::forget_all());
                            level_id.set(LEVELS[0].id.to_string());
                        }
                    }
                >
                    "RESET PROGRESS"
                </button>
                <span class="status" title="The build you are running: quote it when reporting a problem">
                    {concat!("MILESTONE 1 · BUILD ", env!("BUILD_COMMIT"))}
                </span>
            </footer>
        </div>
    }
}

/// The RL loop's REWARD value: one move's reward.
fn reward_text(reward: Option<f32>) -> String {
    reward.map_or("—".into(), |r| format!("{} · this move", signed(f64::from(r))))
}

/// The RL loop's RETURN value: the episode's rewards so far, added up.
fn return_text(episode_return: f32, moves: u32) -> String {
    match moves {
        0 => "0 · no rewards yet".into(),
        1 => format!("{} · 1 reward so far", signed(f64::from(episode_return))),
        n => format!("{} · sum of {n} rewards", signed(f64::from(episode_return))),
    }
}
