//! CONFIGURE, TRAIN and DEPLOY views for agent-piloted levels.

use leptos::prelude::*;
use rl_agents::QLearningConfig;
use rl_tutorial::UiElement;

use crate::analyse::{EpisodeChart, Hover, episode_chart};
use crate::campaign::Progress;
use crate::chart_math::signed;
use crate::protocol::{
    DeployReport, EpisodeOutcome, ExperimentConfig, FeatureDef, ObservationStatus, PlayerEdits, Snapshot,
    TrainingStatus,
};

/// Episodes averaged by the learning curve's trend line.
pub const CURVE_WINDOW: usize = 50;

/// Settings being edited, applied to a new raft with APPLY.
#[derive(Clone, Debug, PartialEq)]
pub struct Draft {
    pub config: QLearningConfig,
    pub rewards: Vec<(String, f32)>,
    /// The observation features, on levels where the player chooses them.
    pub observation: Option<Vec<FeatureDef>>,
    /// The training wind's slip, on levels where the player sets it.
    pub wind: Option<f32>,
    pub training_seed: u64,
}

impl Draft {
    pub fn from_status(t: &TrainingStatus) -> Self {
        Self {
            config: t.config,
            rewards: t.rewards.iter().map(|r| (r.term.clone(), r.weight)).collect(),
            observation: t.observation.as_ref().map(|o| o.features.clone()),
            wind: t.wind.map(|w| w.slip),
            training_seed: t.training_seed,
        }
    }

    /// An experiment with these settings on `level_id`. Only the reward terms
    /// the level lets the player edit are part of it.
    pub fn experiment(
        &self,
        level_id: &str,
        status: &TrainingStatus,
        episodes: u32,
    ) -> Result<ExperimentConfig, String> {
        let source = crate::content::level(level_id).ok_or("unknown level")?.level_source;
        let editable = |term: &str| status.rewards.iter().any(|r| r.term == term && r.editable);
        let edits = PlayerEdits {
            rewards: self.rewards.iter().filter(|(t, _)| editable(t)).cloned().collect(),
            observation: self.observation.clone(),
            wind: self.wind,
        };
        ExperimentConfig::q_learning(level_id, source, edits, self.config, self.training_seed, episodes)
            .map_err(|e| crate::sim::errors_text(&e))
    }

    /// Whether the draft differs from what the current raft was trained with.
    pub fn differs_from(&self, t: &TrainingStatus) -> bool {
        *self != Self::from_status(t)
    }
}

#[derive(Clone, Copy)]
pub struct TrainCtx {
    pub snapshot: RwSignal<Option<Snapshot>>,
    pub progress: RwSignal<Progress>,
    pub highlights: Memo<Vec<UiElement>>,
    pub draft: RwSignal<Option<Draft>>,
    pub training: RwSignal<bool>,
    pub deploy: RwSignal<Option<DeployReport>>,
    pub apply: Callback<()>,
    pub train: Callback<u32>,
    pub stop_training: Callback<()>,
    pub open_walkthrough: Callback<()>,
    pub deploy_now: Callback<()>,
    pub watch_trained: Callback<u64>,
    /// Runs the draft settings as an experiment, keeping the current raft.
    pub run_experiment: Callback<()>,
}

impl TrainCtx {
    fn status(&self) -> Option<TrainingStatus> {
        self.snapshot.with(|s| s.as_ref().and_then(|s| s.training.clone()))
    }

    fn hl(&self, el: UiElement) -> bool {
        self.highlights.with(|h| h.contains(&el))
    }
}

fn number_input(
    value: impl Fn() -> f32 + Send + Sync + 'static,
    set: impl Fn(f32) + 'static,
    (lo, hi): (f32, f32),
    step: &'static str,
) -> impl IntoView {
    view! {
        <input
            type="number"
            min=lo.to_string()
            max=hi.to_string()
            step=step
            prop:value=move || format!("{}", value())
            on:change=move |e| {
                if let Ok(v) = event_target_value(&e).parse::<f32>() {
                    set(v.clamp(lo, hi));
                }
            }
        />
    }
}

pub fn configure_view(ctx: TrainCtx) -> impl IntoView {
    let setting = move |name: &'static str,
                        symbol: &'static str,
                        about: &'static str,
                        get: fn(&QLearningConfig) -> f32,
                        put: fn(&mut QLearningConfig, f32),
                        bounds: (f32, f32)| {
        view! {
            <div class="setting">
                <span class="setting-name">{format!("{symbol}  {name}")}</span>
                {number_input(
                    move || ctx.draft.with(|d| d.as_ref().map_or(0.0, |d| get(&d.config))),
                    move |v| ctx.draft.update(|d| if let Some(d) = d { put(&mut d.config, v) }),
                    bounds,
                    "0.01",
                )}
                <span class="setting-range">{format!("{} to {}", bounds.0, bounds.1)}</span>
                <span class="setting-about">{about}</span>
            </div>
        }
    };
    move || {
        let Some(status) = ctx.status() else {
            return view! { <p class="placeholder">"This level has no trainable agent."</p> }.into_any();
        };
        let b = status.bounds;
        let dirty = move || ctx.draft.with(|d| d.as_ref().zip(ctx.status()).is_some_and(|(d, t)| d.differs_from(&t)));
        view! {
            <section class="training-view" class:hl-box=move || ctx.hl(UiElement::Configure)>
                <h2>"CONFIGURE · A NEW RAFT"</h2>
                <p class="subtitle">
                    {if status.observation.is_some() {
                        "Settings apply to a fresh raft, its Q-table all zeros.".to_string()
                    } else {
                        format!("Settings apply to a fresh raft. Its Q-table holds {} numbers, all starting at zero.", status.table_size)
                    }}
                </p>
                {status.observation.clone().map(|o| observation_builder(ctx, o))}
                {status.wind.map(|w| {
                    view! {
                        <div class="wind-setting" class:hl-box=move || ctx.hl(UiElement::Wind)>
                            <h3>"TRAINING WIND"</h3>
                            <div class="setting">
                                <span class="setting-name">"slip in windy water"</span>
                                {number_input(
                                    move || ctx.draft.with(|d| d.as_ref().and_then(|d| d.wind).unwrap_or(w.slip)),
                                    move |v| ctx.draft.update(|d| if let Some(d) = d { d.wind = Some(v) }),
                                    w.range,
                                    "0.05",
                                )}
                                <span class="setting-range">{format!("{} to {}", w.range.0, w.range.1)}</span>
                                <span class="setting-about">
                                    {format!(
                                        "the chance a move in windy water is turned aside while training; tests always sail the real wind, {:.0}%",
                                        w.level * 100.0
                                    )}
                                </span>
                            </div>
                        </div>
                    }
                })}
                <h3>"LEARNING"</h3>
                {setting("learning rate", "α", "how far each update moves a value toward its target", |c| c.alpha, |c, v| c.alpha = v, b.alpha)}
                {setting("discount", "γ", "how much later rewards count compared with immediate ones", |c| c.gamma, |c, v| c.gamma = v, b.gamma)}
                {setting("exploration", "ε", "the chance of a random move while training", |c| c.epsilon, |c, v| c.epsilon = v, b.epsilon)}
                <div class="rewards" class:hl-box=move || ctx.hl(UiElement::Rewards)>
                    <h3>"REWARDS"</h3>
                    {status
                        .reward_range
                        .map(|(lo, hi)| view! { <p class="subtitle">{format!("Editable rewards range from {} to {}.", signed(f64::from(lo)), signed(f64::from(hi)))}</p> })}
                    {status
                        .rewards
                        .iter()
                        .map(|r| {
                            let term = r.term.clone();
                            let range = status.reward_range.unwrap_or((r.weight, r.weight));
                            let get_term = term.clone();
                            let value = move || {
                                ctx.draft.with(|d| {
                                    d.as_ref().and_then(|d| d.rewards.iter().find(|(t, _)| *t == get_term).map(|(_, w)| *w)).unwrap_or(0.0)
                                })
                            };
                            let set_term = term.clone();
                            let set = move |v: f32| {
                                ctx.draft.update(|d| {
                                    if let Some(d) = d && let Some(entry) = d.rewards.iter_mut().find(|(t, _)| *t == set_term) {
                                        entry.1 = v;
                                    }
                                })
                            };
                            view! {
                                <div class="setting">
                                    <span class="setting-name">{r.label.clone()}</span>
                                    {if r.editable {
                                        number_input(value, set, range, "0.5").into_any()
                                    } else {
                                        view! { <span class="fixed">{signed(f64::from(r.weight))} " · fixed"</span> }.into_any()
                                    }}
                                </div>
                            }
                        })
                        .collect_view()}
                </div>
                <h3>"TRAINING SEED"</h3>
                <div class="setting">
                    <span class="setting-name">"seed"</span>
                    <input
                        type="number"
                        min="0"
                        prop:value=move || ctx.draft.with(|d| d.as_ref().map_or(0, |d| d.training_seed)).to_string()
                        on:change=move |e| {
                            if let Ok(v) = event_target_value(&e).parse::<u64>() {
                                ctx.draft.update(|d| if let Some(d) = d { d.training_seed = v });
                            }
                        }
                    />
                    <span></span>
                    <span class="setting-about">"the same seed and settings always train the same raft"</span>
                </div>
                <div class="row actions">
                    <button class="primary" on:click=move |_| ctx.apply.run(())>"APPLY · START A NEW RAFT"</button>
                    <button
                        title="Train a separate raft with these settings and add it to COMPARE; your current raft is kept"
                        on:click=move |_| ctx.run_experiment.run(())
                    >
                        "RUN AS EXPERIMENT ▶"
                    </button>
                    {move || dirty().then(|| view! { <span class="warning">"unapplied changes"</span> })}
                </div>
            </section>
        }
        .into_any()
    }
}

/// The Observation Builder: which features the raft observes, and what each
/// choice does to the Q-table's size.
fn observation_builder(ctx: TrainCtx, o: ObservationStatus) -> impl IntoView {
    let allowed = o.allowed.clone();
    let chosen = move || ctx.draft.with(|d| d.as_ref().and_then(|d| d.observation.clone()).unwrap_or_default());
    let rows = {
        let allowed = allowed.clone();
        move || {
            let chosen = chosen();
            let parts: Vec<u32> =
                allowed.iter().filter(|f| chosen.contains(&f.feature)).map(|f| f.cardinality).collect();
            let total: u64 = parts.iter().map(|&c| u64::from(c)).product();
            (parts, total)
        }
    };
    let toggle = {
        let allowed = allowed.clone();
        move |feature: FeatureDef| {
            ctx.draft.update(|d| {
                let Some(d) = d else { return };
                let mut now = d.observation.clone().unwrap_or_default();
                if let Some(i) = now.iter().position(|f| *f == feature) {
                    if now.len() > 1 {
                        now.remove(i);
                    }
                } else {
                    now.push(feature);
                }
                // Keep the level's order, so the same choice always gives the same table.
                d.observation = Some(allowed.iter().map(|a| a.feature.clone()).filter(|f| now.contains(f)).collect());
            })
        }
    };
    view! {
        <div class="observation" class:hl-box=move || ctx.hl(UiElement::Observation)>
            <h3>"OBSERVATION · WHAT THE RAFT SEES"</h3>
            <p class="subtitle">"The Q-table has one row for every different thing the raft can observe."</p>
            {allowed
                .into_iter()
                .map(|f| {
                    let feature = f.feature.clone();
                    let is_on = {
                        let feature = feature.clone();
                        move || chosen().contains(&feature)
                    };
                    let last = {
                        let is_on = is_on.clone();
                        move || is_on() && chosen().len() == 1
                    };
                    let toggle = toggle.clone();
                    view! {
                        <label class="setting feature">
                            <input
                                type="checkbox"
                                prop:checked=is_on
                                disabled=last
                                on:change=move |_| toggle(feature.clone())
                            />
                            <span class="setting-name">{f.name.clone()}</span>
                            <span class="setting-range">{format!("× {} values", f.cardinality)}</span>
                        </label>
                    }
                })
                .collect_view()}
            <p class="table-size">
                {move || {
                    let (parts, total) = rows();
                    let product = parts.iter().map(u32::to_string).collect::<Vec<_>>().join(" × ");
                    let product = if parts.len() > 1 { format!("{product} = ") } else { String::new() };
                    format!("Q-table: {product}{total} rows × 4 moves = {} numbers", total * 4)
                }}
            </p>
        </div>
    }
}

pub fn train_view(ctx: TrainCtx) -> impl IntoView {
    let hover = RwSignal::new(None::<Hover>);
    move || {
        let Some(status) = ctx.status() else {
            return view! { <p class="placeholder">"This level has no trainable agent."</p> }.into_any();
        };
        let used = status.episodes_trained;
        let budget = status.budget;
        let left = budget.map(|b| b.saturating_sub(used));
        let c = status.config;
        let reward_summary = status
            .rewards
            .iter()
            .map(|r| format!("{} {}", r.label, signed(f64::from(r.weight))))
            .collect::<Vec<_>>()
            .join(" · ");
        let curve = status.curve.clone();
        let recent = &curve[curve.len().saturating_sub(CURVE_WINDOW)..];
        let reach = if recent.is_empty() {
            0.0
        } else {
            recent.iter().filter(|e| e.reached_goal).count() as f32 / recent.len() as f32
        };
        let trend = rl_training::moving_mean(&curve, CURVE_WINDOW);
        let goal = ctx.snapshot.with(|s| s.as_ref().map(|s| s.goal_name.clone())).unwrap_or_default();
        view! {
            <section class="training-view" class:hl-box=move || ctx.hl(UiElement::Train)>
                <h2>"TRAIN"</h2>
                <p class="subtitle">
                    {format!("α {} · γ {} · ε {} · {reward_summary}", c.alpha, c.gamma, c.epsilon)}
                </p>
                <div class="budget">
                    <span>{format!("{used} episodes trained")}</span>
                    {budget.map(|b| {
                        let fraction = f64::from(used) / f64::from(b.max(1));
                        view! {
                            <span class="budget-bar"><span class="budget-fill" style=format!("width: {:.1}%", fraction * 100.0)></span></span>
                            <span>{format!("budget {b} · {} left", b.saturating_sub(used))}</span>
                        }
                    })}
                </div>
                <div class="row actions">
                    {[100u32, 500]
                        .into_iter()
                        .map(|n| {
                            view! {
                                <button
                                    disabled=move || ctx.training.get() || left == Some(0)
                                    on:click=move |_| ctx.train.run(n)
                                >
                                    {format!("TRAIN {n}")}
                                </button>
                            }
                        })
                        .collect_view()}
                    <button
                        class="primary"
                        disabled=move || ctx.training.get() || left == Some(0)
                        on:click=move |_| ctx.train.run(left.unwrap_or(1_000))
                    >
                        "TRAIN TO BUDGET"
                    </button>
                    {move || ctx.training.get().then(|| view! { <button on:click=move |_| ctx.stop_training.run(())>"■ STOP"</button> })}
                </div>
                <div class="row actions" class:hl-box=move || ctx.hl(UiElement::Walkthrough)>
                    <button disabled=move || ctx.training.get() on:click=move |_| ctx.open_walkthrough.run(())>
                        "STEP THROUGH ▶"
                    </button>
                    <span class="subtitle">"watch Q-learning run line by line, with your raft's real numbers"</span>
                </div>
                {if curve.is_empty() {
                    view! { <p class="placeholder">"No training yet. Each episode the raft sails, it updates its Q-table from the rewards it receives."</p> }.into_any()
                } else {
                    let last = *curve.last().unwrap();
                    view! {
                        <div class="tiles">
                            <div class="tile">
                                <div class="tile-label">"LAST EPISODE"</div>
                                <div class="tile-value">{signed(f64::from(last.episode_return))}</div>
                                <div class="tile-note">{format!("{} moves · {}", last.length, if last.reached_goal { format!("reached {goal}") } else { "did not arrive".into() })}</div>
                            </div>
                            <div class="tile">
                                <div class="tile-label">{format!("LAST {} EPISODES", recent.len())}</div>
                                <div class="tile-value">{format!("{:.0}%", reach * 100.0)}</div>
                                <div class="tile-note">{format!("reached {goal} while still exploring")}</div>
                            </div>
                        </div>
                        <h3>"LEARNING CURVE"</h3>
                        {episode_chart(
                            EpisodeChart {
                                episodes: &curve,
                                trend,
                                trend_label: "last-50 mean",
                                unit: "training episode",
                                best_possible: None,
                                mark_best: None,
                                aria: "Return of every training episode, with a 50-episode moving average",
                            },
                            hover,
                        )}
                        <p class="subtitle">"Training returns include exploration: with ε above zero, some moves are random on purpose. DEPLOY shows what the raft does without exploring."</p>
                    }
                    .into_any()
                }}
            </section>
        }
        .into_any()
    }
}

pub fn deploy_view(ctx: TrainCtx) -> impl IntoView {
    move || {
        let Some(status) = ctx.status() else {
            return view! { <p class="placeholder">"This level has no trainable agent."</p> }.into_any();
        };
        let labels = ctx.snapshot.with(|s| {
            s.as_ref()
                .and_then(|s| {
                    let goal = &s.objective.as_ref()?.goal;
                    Some(ctx.progress.with(|p| p.task_states(&s.level_id, goal)))
                })
                .unwrap_or_default()
        });
        view! {
            <section class="training-view" class:hl-box=move || ctx.hl(UiElement::Deploy)>
                <h2>"DEPLOY"</h2>
                <p class="subtitle">
                    "The trained raft sails each test seed greedily: no exploration, always the move it values most. Test seeds are fixed, so results are reproducible."
                </p>
                <div class="row actions">
                    <button
                        class="primary"
                        disabled=status.episodes_trained == 0
                        on:click=move |_| ctx.deploy_now.run(())
                    >
                        "DEPLOY ON TEST SEEDS"
                    </button>
                    {(status.episodes_trained == 0).then(|| view! { <span class="warning">"train the raft first"</span> })}
                </div>
                {move || {
                    ctx.deploy
                        .get()
                        .map(|r| {
                            let first_seed = r.voyages.first().map_or(0, |v| v.seed);
                            view! {
                                <h3>"RESULTS"</h3>
                                <p class="subtitle">
                                    {format!("after {} training episodes", r.episodes_trained)}
                                    {r.stars.map(|s| format!(" · {} {}", "★".repeat(s as usize) + &"☆".repeat(3usize.saturating_sub(s as usize)), if s == 0 { "over budget" } else { "for training efficiency" }))}
                                </p>
                                <ul class="results">
                                    {r.tasks
                                        .iter()
                                        .map(|t| {
                                            let label = labels.get(t.task).map_or(String::new(), |l| l.title.clone());
                                            view! {
                                                <li class:passed=t.passed>
                                                    {format!(
                                                        "{} {} · {} of {} test voyages (needs {})",
                                                        if t.passed { "✓" } else { "✗" },
                                                        label,
                                                        t.successes,
                                                        t.tested,
                                                        t.needed
                                                    )}
                                                </li>
                                            }
                                        })
                                        .collect_view()}
                                </ul>
                                <table class="stats">
                                    <thead>
                                        <tr>
                                            <th scope="col">"seed"</th>
                                            <th scope="col">"outcome"</th>
                                            <th scope="col">"return"</th>
                                            <th scope="col">"moves"</th>
                                            <th scope="col">"on rocks"</th>
                                            <th scope="col">"in lotus"</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {r.voyages
                                            .iter()
                                            .map(|v| {
                                                let outcome = match v.outcome {
                                                    EpisodeOutcome::ReachedGoal => "arrived",
                                                    EpisodeOutcome::Died => "whirlpool",
                                                    EpisodeOutcome::Caught => "caught",
                                                    EpisodeOutcome::OutOfTime => "out of time",
                                                };
                                                view! {
                                                    <tr>
                                                        <td>{v.seed.to_string()}</td>
                                                        <td>{outcome}</td>
                                                        <td>{signed(f64::from(v.episode_return))}</td>
                                                        <td>{v.length.to_string()}</td>
                                                        <td>{v.hazard_steps.to_string()}</td>
                                                        <td>{v.lure_steps.to_string()}</td>
                                                    </tr>
                                                }
                                            })
                                            .collect_view()}
                                    </tbody>
                                </table>
                                <div class="row actions">
                                    <button class="primary" on:click=move |_| ctx.watch_trained.run(first_seed)>
                                        "WATCH THE TRAINED RAFT ▶"
                                    </button>
                                </div>
                            }
                        })
                }}
            </section>
        }
        .into_any()
    }
}
