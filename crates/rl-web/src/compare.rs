//! The COMPARE view: run experiments from fixed settings, rerun them to check
//! they reproduce, and compare up to four side by side.
//!
//! The chart follows the project's chart rules: one validated colour per
//! series in fixed slot order, plus a dash pattern and a letter, so no series
//! depends on colour alone; a legend and direct labels; a hover tooltip; and a
//! table with every number.

use std::collections::BTreeMap;

use leptos::prelude::*;
use rl_training::{EpisodeOutcome, moving_mean};
use rl_tutorial::UiElement;
use wasm_bindgen::JsCast;

use crate::analyse::{H, Hover, PAD_L, PAD_T, PLOT_H, PLOT_W, W, tooltip, y_axis};
use crate::chart_math::{nice_step, signed};
use crate::experiments::{COMPARE_WINDOW, ExperimentLog, LoggedRun, SLOT_LETTERS, Selection, settings_line};
use crate::protocol::Snapshot;
use crate::training_ui::Draft;

/// What a pending run is for.
#[derive(Clone, Debug, PartialEq)]
pub enum Pending {
    /// A new run for the log.
    New { started_ms: f64 },
    /// Rerunning a logged run to check that it reproduces.
    Rerun { id: u32 },
    /// Rebuilding a logged run's raft to make it the current raft.
    Adopt { id: u32 },
}

#[derive(Clone, Copy)]
pub struct CompareCtx {
    pub snapshot: RwSignal<Option<Snapshot>>,
    pub log: RwSignal<ExperimentLog>,
    pub selection: RwSignal<Selection>,
    pub pending: RwSignal<Option<Pending>>,
    /// Rerun results: whether each rerun matched the logged fingerprint.
    pub reruns: RwSignal<BTreeMap<u32, bool>>,
    pub draft: RwSignal<Option<Draft>>,
    pub episodes: RwSignal<u32>,
    pub highlights: Memo<Vec<UiElement>>,
    pub run: Callback<()>,
    pub rerun: Callback<u32>,
    pub adopt: Callback<u32>,
    pub remove: Callback<u32>,
    pub watch: Callback<(u32, usize)>,
    pub edit_settings: Callback<()>,
}

/// Dash pattern per slot: the series' second channel after colour.
const DASHES: [&str; 4] = ["none", "7 4", "2 3", "9 3 2 3"];

fn percent(v: f32) -> String {
    format!("{:.0}%", v * 100.0)
}

pub fn compare_view(ctx: CompareCtx) -> impl IntoView {
    let hover = RwSignal::new(None::<Hover>);
    let budget = move || ctx.snapshot.with(|s| s.as_ref().and_then(|s| s.budget_episodes));
    let draft_line = move || {
        ctx.snapshot.with(|s| {
            let s = s.as_ref()?;
            let status = s.training.as_ref()?;
            ctx.draft.with(|d| {
                d.as_ref().map(|d| match d.experiment(&s.level_id, status, ctx.episodes.get()) {
                    Ok(c) => settings_line(&c),
                    Err(e) => e,
                })
            })
        })
    };
    let busy = move || ctx.pending.with(Option::is_some);

    view! {
        <section class="training-view compare" class:hl-box=move || ctx.highlights.with(|h| h.contains(&UiElement::Compare))>
            <h2>"COMPARE · EXPERIMENTS"</h2>
            <p class="subtitle">
                "An experiment trains a new raft from fixed settings, then tests it on the level's test seeds. Your current raft is not touched. The same settings always give the same result: RERUN checks it."
            </p>
            <h3>"NEW EXPERIMENT"</h3>
            <p class="settings-line">{move || draft_line().unwrap_or_else(|| "no trainable agent in this level".into())}</p>
            <div class="row actions">
                <button on:click=move |_| ctx.edit_settings.run(())>"EDIT SETTINGS IN CONFIGURE"</button>
                <label class="setting-inline">
                    "episodes "
                    <input
                        type="number"
                        min="1"
                        max=move || budget().map(|b| b.to_string())
                        prop:value=move || ctx.episodes.get().to_string()
                        on:change=move |e| {
                            if let Ok(v) = event_target_value(&e).parse::<u32>() {
                                ctx.episodes.set(v.clamp(1, budget().unwrap_or(20_000)));
                            }
                        }
                    />
                </label>
                <button class="primary" disabled=busy on:click=move |_| ctx.run.run(())>
                    {move || if busy() { "RUNNING…" } else { "RUN EXPERIMENT ▶" }}
                </button>
            </div>
            {move || log_table(ctx)}
            {move || comparison(ctx, hover)}
        </section>
    }
}

fn log_table(ctx: CompareCtx) -> impl IntoView {
    let runs = ctx.log.with(|l| l.runs.clone());
    if runs.is_empty() {
        return view! {
            <p class="placeholder">"No experiments yet. Run one, change a setting in CONFIGURE, run another, then compare them."</p>
        }
        .into_any();
    }
    let busy = move || ctx.pending.with(Option::is_some);
    view! {
        <h3>{format!("LOG · {} RUNS", runs.len())}</h3>
        <p class="subtitle">"Tick up to four runs to compare them."</p>
        <table class="stats log">
            <thead>
                <tr>
                    <th scope="col">"compare"</th>
                    <th scope="col">"run"</th>
                    <th scope="col">"settings"</th>
                    <th scope="col">"test voyages"</th>
                    <th scope="col">"fingerprint"</th>
                    <th scope="col"></th>
                </tr>
            </thead>
            <tbody>
                {runs
                    .into_iter()
                    .rev()
                    .map(|r| {
                        let id = r.id;
                        let slot = move || ctx.selection.with(|s| s.slot(id));
                        let e = &r.evaluation;
                        let result = format!("{} of {} arrived", e.reached, e.tested());
                        view! {
                            <tr>
                                <td>
                                    <label class="pick">
                                        <input
                                            type="checkbox"
                                            prop:checked=move || slot().is_some()
                                            disabled=move || slot().is_none() && ctx.selection.with(Selection::is_full)
                                            on:change=move |_| {
                                                ctx.selection.update(|s| {
                                                    s.toggle(id);
                                                })
                                            }
                                        />
                                        {move || slot().map(|i| view! { <span class=format!("slot-key s{i}")>{SLOT_LETTERS[i]}</span> })}
                                    </label>
                                </td>
                                <th scope="row">{r.label.clone()}</th>
                                <td class="left">{r.settings()}</td>
                                <td>{result}</td>
                                <td>
                                    <code>{r.short_fingerprint()}</code>
                                    {move || {
                                        ctx.reruns
                                            .with(|m| m.get(&id).copied())
                                            .map(|same| {
                                                view! {
                                                    <span class:match=same class:mismatch=!same>
                                                        {if same { " ✓ reproduced exactly" } else { " ✗ differs" }}
                                                    </span>
                                                }
                                            })
                                    }}
                                </td>
                                <td class="actions-cell">
                                    <button class="quiet" disabled=busy title="Run the same settings again and compare the fingerprint" on:click=move |_| ctx.rerun.run(id)>
                                        "RERUN"
                                    </button>
                                    <button class="quiet" disabled=busy title="Make this run's raft your current raft" on:click=move |_| ctx.adopt.run(id)>
                                        "ADOPT"
                                    </button>
                                    <button class="quiet" title="Remove from the log" aria-label="Remove from the log" on:click=move |_| ctx.remove.run(id)>
                                        "✕"
                                    </button>
                                </td>
                            </tr>
                        }
                    })
                    .collect_view()}
            </tbody>
        </table>
    }
    .into_any()
}

/// The selected runs, in slot order.
fn selected(ctx: CompareCtx) -> Vec<(usize, LoggedRun)> {
    let selection = ctx.selection.get();
    ctx.log.with(|l| selection.slots().filter_map(|(i, id)| l.get(id).map(|r| (i, r.clone()))).collect())
}

fn comparison(ctx: CompareCtx, hover: RwSignal<Option<Hover>>) -> impl IntoView {
    let runs = selected(ctx);
    if runs.is_empty() {
        return ().into_any();
    }
    view! {
        <h3>"LEARNING CURVES"</h3>
        {curves_chart(&runs, hover)}
        <h3>"SIDE BY SIDE"</h3>
        {metrics_table(&runs)}
        <h3>"TEST VOYAGES"</h3>
        <p class="subtitle">"Each voyage is stored as its seed and its moves. WATCH plays it back on the arena exactly."</p>
        {voyages(ctx, &runs)}
    }
    .into_any()
}

/// One line per run: the moving mean of the training return.
fn curves_chart(runs: &[(usize, LoggedRun)], hover: RwSignal<Option<Hover>>) -> impl IntoView {
    let series: Vec<(usize, Vec<f32>)> =
        runs.iter().map(|(slot, r)| (*slot, moving_mean(&r.curve, COMPARE_WINDOW))).collect();
    let n = series.iter().map(|(_, v)| v.len()).max().unwrap_or(1).max(1);
    let all = series.iter().flat_map(|(_, v)| v.iter().map(|&x| f64::from(x)));
    let lo = all.clone().fold(f64::INFINITY, f64::min).min(0.0);
    let hi = all.fold(f64::NEG_INFINITY, f64::max).max(0.0);
    let pad = ((hi - lo) * 0.08).max(0.5);
    let (lo, hi) = (lo - pad, hi + pad);
    let x = move |i: usize| PAD_L + if n > 1 { i as f64 / (n - 1) as f64 * PLOT_W } else { PLOT_W / 2.0 };
    let y = move |v: f64| PAD_T + (hi - v) / (hi - lo) * PLOT_H;

    let lines = series
        .iter()
        .map(|(slot, values)| {
            let points =
                values.iter().enumerate().map(|(i, v)| format!("{:.1},{:.1}", x(i), y(f64::from(*v)))).collect::<Vec<_>>();
            let end = values.last().map(|&v| (x(values.len() - 1), y(f64::from(v)), v));
            view! {
                <polyline class=format!("series curve s{slot}") points=points.join(" ") stroke-dasharray=DASHES[*slot]></polyline>
                {end.map(|(ex, ey, v)| view! {
                    <text class="label" x=ex + 6.0 y=ey + 4.0>{format!("{} {}", SLOT_LETTERS[*slot], signed(f64::from(v)))}</text>
                })}
            }
        })
        .collect_view();
    let x_ticks: Vec<usize> = {
        let step = nice_step(n as f64, 4).max(1.0) as usize;
        (0..n).step_by(step).chain(std::iter::once(n - 1)).collect()
    };
    let for_hover = series.clone();
    let on_move = move |ev: web_sys::MouseEvent| {
        let Some(target) = ev.current_target().and_then(|t| t.dyn_into::<web_sys::Element>().ok()) else {
            return;
        };
        let rect = target.get_bounding_client_rect();
        let frac = ((f64::from(ev.client_x()) - rect.left()) / rect.width()).clamp(0.0, 1.0);
        let i = (frac * (n - 1) as f64).round() as usize;
        let parts: Vec<String> = for_hover
            .iter()
            .filter_map(|(slot, v)| v.get(i).map(|m| format!("{} {}", SLOT_LETTERS[*slot], signed(f64::from(*m)))))
            .collect();
        let top = for_hover.iter().filter_map(|(_, v)| v.get(i)).fold(f64::NEG_INFINITY, |a, &b| a.max(f64::from(b)));
        hover.set(Some(Hover {
            x: x(i) / W * 100.0,
            y: y(top) / H * 100.0,
            text: format!("episode {} · last-{COMPARE_WINDOW} mean · {}", i + 1, parts.join(" · ")),
        }));
    };

    view! {
        <div class="chart-wrap">
            <svg class="chart" viewBox=format!("0 0 {W} {H}") role="img" aria-label="Learning curves of the compared runs: the 50-episode moving mean of the training return">
                {y_axis(lo, hi, y)}
                {x_ticks
                    .into_iter()
                    .map(|i| view! { <text class="tick" x=x(i) y=H - 8.0 text-anchor="middle">{(i + 1).to_string()}</text> })
                    .collect_view()}
                {lines}
                {move || {
                    hover.get().map(|h| {
                        let px = h.x / 100.0 * W;
                        view! { <line class="crosshair" x1=px x2=px y1=PAD_T y2=PAD_T + PLOT_H></line> }
                    })
                }}
                <rect class="hit" x=PAD_L y=PAD_T width=PLOT_W height=PLOT_H on:mousemove=on_move on:mouseleave=move |_| hover.set(None)></rect>
            </svg>
            {tooltip(hover)}
        </div>
        <div class="legend">
            {runs
                .iter()
                .map(|(slot, r)| {
                    view! {
                        <svg class="series-key" viewBox="0 0 22 6" aria-hidden="true">
                            <line class=format!("series s{slot}") x1="0" x2="22" y1="3" y2="3" stroke-dasharray=DASHES[*slot]></line>
                        </svg>
                        <span>{format!("{} · {}", SLOT_LETTERS[*slot], r.label)}</span>
                    }
                })
                .collect_view()}
            <span class="subtitle">{format!("· training return, last-{COMPARE_WINDOW} mean · x axis: training episode")}</span>
        </div>
    }
}

fn metrics_table(runs: &[(usize, LoggedRun)]) -> impl IntoView {
    type Row = (&'static str, fn(&LoggedRun) -> String);
    // Rows that only matter on some maps are left out when every run shows nothing in them.
    const OPTIONAL: [&str; 4] = ["by the shortest route", "lost to a whirlpool", "caught", "out of time"];
    let rows: Vec<Row> = vec![
        ("settings", |r| settings_line(&r.config)),
        ("training episodes", |r| r.curve.len().to_string()),
        ("training moves", |r| r.training_steps().to_string()),
        ("random moves (exploration)", |r| percent(r.exploration_rate())),
        ("final training mean", |r| {
            moving_mean(&r.curve, COMPARE_WINDOW).last().map_or("—".into(), |m| signed(f64::from(*m)))
        }),
        ("test voyages arrived", |r| format!("{} of {}", r.evaluation.reached, r.evaluation.tested())),
        ("by the shortest route", |r| r.evaluation.shortest.map_or("—".into(), |s| s.to_string())),
        ("lost to a whirlpool", |r| r.evaluation.died.to_string()),
        ("caught", |r| r.evaluation.caught.to_string()),
        ("out of time", |r| r.evaluation.out_of_time.to_string()),
        ("mean test return", |r| signed(f64::from(r.evaluation.mean_return))),
        ("mean test moves", |r| format!("{:.1}", r.evaluation.mean_length)),
        ("time to run", |r| r.wall_clock_ms.map_or("—".into(), |ms| format!("{:.0} ms", ms))),
        ("fingerprint", |r| r.short_fingerprint()),
        ("engine", |r| r.engine.clone()),
    ];
    let rows: Vec<Row> = rows
        .into_iter()
        .filter(|(name, f)| !OPTIONAL.contains(name) || runs.iter().any(|(_, r)| !matches!(f(r).as_str(), "0" | "—")))
        .collect();
    view! {
        <table class="stats side-by-side">
            <thead>
                <tr>
                    <th scope="col"></th>
                    {runs
                        .iter()
                        .map(|(slot, r)| view! { <th scope="col"><span class=format!("slot-key s{slot}")>{SLOT_LETTERS[*slot]}</span>" "{r.label.clone()}</th> })
                        .collect_view()}
                </tr>
            </thead>
            <tbody>
                {rows
                    .into_iter()
                    .map(|(name, f)| {
                        view! {
                            <tr>
                                <th scope="row">{name}</th>
                                {runs.iter().map(|(_, r)| view! { <td>{f(r)}</td> }).collect_view()}
                            </tr>
                        }
                    })
                    .collect_view()}
            </tbody>
        </table>
    }
}

fn voyages(ctx: CompareCtx, runs: &[(usize, LoggedRun)]) -> impl IntoView {
    runs.iter()
        .map(|(slot, r)| {
            let id = r.id;
            view! {
                <div class="voyage-row">
                    <span class=format!("slot-key s{slot}")>{SLOT_LETTERS[*slot]}</span>
                    {r.evaluation
                        .episodes
                        .iter()
                        .enumerate()
                        .map(|(k, e)| {
                            let p = e.played;
                            let (mark, outcome) = match p.outcome {
                                EpisodeOutcome::ReachedGoal => ("✓", "arrived"),
                                EpisodeOutcome::Died => ("✗", "lost"),
                                EpisodeOutcome::Caught => ("✗", "caught"),
                                EpisodeOutcome::OutOfTime => ("…", "out of time"),
                            };
                            let about = format!(
                                "seed {} · {outcome} · return {} · {} moves · watch it",
                                p.seed,
                                signed(f64::from(p.episode_return)),
                                p.length
                            );
                            view! {
                                <button class="voyage-chip" class:failed=p.outcome != EpisodeOutcome::ReachedGoal title=about.clone() aria-label=about on:click=move |_| ctx.watch.run((id, k))>
                                    {format!("{} {mark}", p.seed)}
                                </button>
                            }
                        })
                        .collect_view()}
                </div>
            }
        })
        .collect_view()
}
