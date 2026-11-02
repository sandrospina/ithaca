//! The ANALYSE view: batch experiments on an agent, their statistics, and
//! the level's questions.
//!
//! Charts follow the project's visual rules: thin marks, one hue per series
//! (agent cyan for voyages, goal gold for the running mean), recessive
//! hairline axes, direct labels on the few marks that matter, a hover
//! tooltip, and a table so no value depends on hovering or on colour.

use std::collections::BTreeMap;
use std::sync::Arc;

use leptos::prelude::*;
use rl_envs::{GoalDef, QuestionDef, TaskDef};
use rl_training::BatchReport;
use rl_tutorial::UiElement;
use wasm_bindgen::JsCast;

use crate::campaign::{Answer, AnswerOutcome, Progress, band_labels, percent};
use crate::chart_math::{nice_step, signed, ticks};
use crate::protocol::Snapshot;

/// Batch sizes offered to the player.
pub const BATCH_SIZES: [u32; 4] = [10, 100, 1_000, 10_000];

const W: f64 = 680.0;
const H: f64 = 230.0;
const PAD_L: f64 = 48.0;
const PAD_R: f64 = 128.0;
const PAD_T: f64 = 14.0;
const PAD_B: f64 = 30.0;
const PLOT_W: f64 = W - PAD_L - PAD_R;
const PLOT_H: f64 = H - PAD_T - PAD_B;

/// Shared state and actions for the view.
#[derive(Clone, Copy)]
pub struct AnalyseCtx {
    pub report: RwSignal<Option<BatchReport>>,
    pub running: RwSignal<bool>,
    pub batch_size: RwSignal<u32>,
    pub first_seed: RwSignal<u64>,
    pub snapshot: RwSignal<Option<Snapshot>>,
    pub progress: RwSignal<Progress>,
    pub feedback: RwSignal<BTreeMap<(usize, usize), AnswerOutcome>>,
    pub highlights: Memo<Vec<UiElement>>,
    pub run_batch: Callback<()>,
    pub replay_best: Callback<u64>,
    pub submit_answer: Callback<(usize, usize, Answer)>,
}

#[derive(Clone, Debug, PartialEq)]
struct Hover {
    /// Position in percent of the chart box.
    x: f64,
    y: f64,
    text: String,
}

fn tooltip(hover: RwSignal<Option<Hover>>) -> impl IntoView {
    move || {
        hover.get().map(|h| {
            view! {
                <div class="tooltip" style=format!("left: {:.2}%; top: {:.2}%", h.x, h.y)>
                    {h.text}
                </div>
            }
        })
    }
}

fn y_axis(lo: f64, hi: f64, y: impl Fn(f64) -> f64) -> impl IntoView {
    ticks(lo, hi, 5)
        .into_iter()
        .map(|t| {
            let py = y(t);
            view! {
                <line class="grid" x1=PAD_L x2=PAD_L + PLOT_W y1=py y2=py></line>
                <text class="tick" x=PAD_L - 6.0 y=py + 4.0 text-anchor="end">{signed(t)}</text>
            }
        })
        .collect_view()
}

/// Scatter of every voyage's return, with the running mean and the best possible return.
fn returns_chart(r: &BatchReport, hover: RwSignal<Option<Hover>>) -> impl IntoView {
    let n = r.episodes.len();
    let best_possible = r.best_possible.map(f64::from);
    let mut lo = f64::from(r.returns.min);
    let mut hi = f64::from(r.returns.max);
    if let Some(b) = best_possible {
        lo = lo.min(b);
        hi = hi.max(b);
    }
    let pad = ((hi - lo) * 0.06).max(1.0);
    let (lo, hi) = (lo - pad, hi + pad);
    let x = move |i: usize| {
        PAD_L
            + if n > 1 {
                i as f64 / (n - 1) as f64 * PLOT_W
            } else {
                PLOT_W / 2.0
            }
    };
    let y = move |v: f64| PAD_T + (hi - v) / (hi - lo) * PLOT_H;
    let radius = if n > 2_000 {
        1.3
    } else if n > 300 {
        2.0
    } else {
        3.0
    };

    let dots = r
        .episodes
        .iter()
        .enumerate()
        .map(
            |(i, e)| view! { <circle cx=x(i) cy=y(f64::from(e.episode_return)) r=radius></circle> },
        )
        .collect_view();
    let mean = r.running_mean();
    let mean_points = mean
        .iter()
        .enumerate()
        .map(|(i, m)| format!("{:.1},{:.1}", x(i), y(f64::from(*m))))
        .collect::<Vec<_>>()
        .join(" ");
    let final_mean = f64::from(*mean.last().unwrap_or(&0.0));
    let best = *r.best_episode();
    let (bx, by) = (x(r.best), y(f64::from(best.episode_return)));

    let x_ticks: Vec<usize> = {
        let step = nice_step(n as f64, 4).max(1.0) as usize;
        (0..n).step_by(step).chain(std::iter::once(n - 1)).collect()
    };
    let episodes = r.episodes.clone();
    let on_move = move |ev: web_sys::MouseEvent| {
        let Some(target) = ev
            .current_target()
            .and_then(|t| t.dyn_into::<web_sys::Element>().ok())
        else {
            return;
        };
        let rect = target.get_bounding_client_rect();
        let frac = ((f64::from(ev.client_x()) - rect.left()) / rect.width()).clamp(0.0, 1.0);
        let i = (frac * (episodes.len() - 1) as f64).round() as usize;
        let e = episodes[i];
        hover.set(Some(Hover {
            x: x(i) / W * 100.0,
            y: y(f64::from(e.episode_return)) / H * 100.0,
            text: format!(
                "voyage {} · seed {} · return {} · {} moves{}",
                i + 1,
                e.seed,
                signed(f64::from(e.episode_return)),
                e.length,
                if e.reached_goal {
                    ""
                } else {
                    " · out of time"
                }
            ),
        }));
    };

    view! {
        <div class="chart-wrap">
            <svg class="chart" viewBox=format!("0 0 {W} {H}") role="img" aria-label="Return of every voyage in the batch, with the running mean">
                {y_axis(lo, hi, y)}
                {x_ticks
                    .into_iter()
                    .map(|i| view! { <text class="tick" x=x(i) y=H - 8.0 text-anchor="middle">{(i + 1).to_string()}</text> })
                    .collect_view()}
                {best_possible
                    .map(|b| {
                        view! {
                            <line class="reference" x1=PAD_L x2=PAD_L + PLOT_W y1=y(b) y2=y(b)></line>
                            <text class="label" x=PAD_L + PLOT_W + 6.0 y=y(b) + 4.0>{format!("best possible {}", signed(b))}</text>
                        }
                    })}
                <g class="series dots">{dots}</g>
                <polyline class="series mean" points=mean_points></polyline>
                <text class="label mean-label" x=PAD_L + PLOT_W + 6.0 y=y(final_mean) + 4.0>
                    {format!("running mean {}", signed(final_mean))}
                </text>
                <circle class="best-ring" cx=bx cy=by r=radius + 4.0></circle>
                <text class="label" x=bx y=by - radius - 8.0 text-anchor="middle">{format!("best · seed {}", best.seed)}</text>
                {move || {
                    hover
                        .get()
                        .map(|h| {
                            let px = h.x / 100.0 * W;
                            view! { <line class="crosshair" x1=px x2=px y1=PAD_T y2=PAD_T + PLOT_H></line> }
                        })
                }}
                <rect
                    class="hit"
                    x=PAD_L
                    y=PAD_T
                    width=PLOT_W
                    height=PLOT_H
                    on:mousemove=on_move
                    on:mouseleave=move |_| hover.set(None)
                ></rect>
            </svg>
            {tooltip(hover)}
        </div>
        <div class="legend">
            <span class="key dot-key"></span>
            "return of one voyage"
            <span class="key line-key"></span>
            "running mean"
            <span class="subtitle">"· x axis: voyage number"</span>
        </div>
    }
}

/// Histogram of returns, with the mean and the best possible return marked.
fn histogram_chart(r: &BatchReport, hover: RwSignal<Option<Hover>>) -> impl IntoView {
    let width = nice_step(f64::from(r.returns.max - r.returns.min), 14).max(1.0);
    let bins = r.histogram(width as f32);
    let n = r.episodes.len();
    let lo = f64::from(bins.first().map_or(0.0, |b| b.0));
    let mut hi = f64::from(bins.last().map_or(0.0, |b| b.0)) + width;
    if let Some(b) = r.best_possible {
        hi = hi.max(f64::from(b) + width);
    }
    let lo = r.best_possible.map_or(lo, |b| lo.min(f64::from(b)));
    let max_count = bins.iter().map(|b| b.1).max().unwrap_or(1).max(1) as f64;
    let x = move |v: f64| PAD_L + (v - lo) / (hi - lo) * PLOT_W;
    let y = move |c: f64| PAD_T + (1.0 - c / max_count) * PLOT_H;
    let mean = f64::from(r.returns.mean);

    let bars = bins
        .iter()
        .map(|&(start, count)| {
            let start = f64::from(start);
            let (x0, x1) = (x(start) + 1.0, x(start + width) - 1.0);
            let top = y(f64::from(count));
            let text = format!(
                "return {} to {} · {} voyages ({})",
                signed(start),
                signed(start + width),
                count,
                percent(count as f32 / n as f32)
            );
            let tip = Hover { x: (x0 + x1) / 2.0 / W * 100.0, y: top / H * 100.0, text };
            view! {
                <rect class="bar" x=x0 y=top width=(x1 - x0).max(0.5) height=PAD_T + PLOT_H - top rx="1"></rect>
                <rect
                    class="hit"
                    x=x0 - 1.0
                    y=PAD_T
                    width=x1 - x0 + 2.0
                    height=PLOT_H
                    on:mouseenter=move |_| hover.set(Some(tip.clone()))
                    on:mouseleave=move |_| hover.set(None)
                ></rect>
            }
        })
        .collect_view();
    let count_ticks = ticks(0.0, max_count, 4)
        .into_iter()
        .map(|t| {
            let py = y(t);
            view! {
                <line class="grid" x1=PAD_L x2=PAD_L + PLOT_W y1=py y2=py></line>
                <text class="tick" x=PAD_L - 6.0 y=py + 4.0 text-anchor="end">{format!("{t}")}</text>
            }
        })
        .collect_view();
    let value_ticks = ticks(lo, hi, 6)
        .into_iter()
        .map(|t| view! { <text class="tick" x=x(t) y=H - 8.0 text-anchor="middle">{signed(t)}</text> })
        .collect_view();

    view! {
        <div class="chart-wrap">
            <svg class="chart" viewBox=format!("0 0 {W} {H}") role="img" aria-label="How the batch's returns are distributed">
                {count_ticks}
                {value_ticks}
                <g class="series">{bars}</g>
                <line class="mean-line" x1=x(mean) x2=x(mean) y1=PAD_T y2=PAD_T + PLOT_H></line>
                <text class="label mean-label" x=x(mean) + 4.0 y=PAD_T + 10.0>{format!("mean {}", signed(mean))}</text>
                {r.best_possible
                    .map(|b| {
                        let px = x(f64::from(b));
                        view! {
                            <line class="reference" x1=px x2=px y1=PAD_T y2=PAD_T + PLOT_H></line>
                            <text class="label" x=px + 4.0 y=PAD_T + 26.0>{format!("best possible {}", signed(f64::from(b)))}</text>
                        }
                    })}
            </svg>
            {tooltip(hover)}
        </div>
        <div class="legend">
            <span class="key bar-key"></span>
            "voyages per return range"
            <span class="key line-key"></span>
            "mean return"
            <span class="subtitle">"· y axis: number of voyages"</span>
        </div>
    }
}

fn stat_table(r: &BatchReport) -> impl IntoView {
    let row = |name: &'static str, s: rl_training::Summary, sign: bool| {
        let f = move |v: f32| {
            if sign {
                signed(f64::from(v))
            } else {
                format!("{v:.1}")
            }
        };
        view! {
            <tr>
                <th scope="row">{name}</th>
                <td>{f(s.min)}</td>
                <td>{f(s.p25)}</td>
                <td>{f(s.median)}</td>
                <td>{f(s.p75)}</td>
                <td>{f(s.max)}</td>
                <td>{f(s.mean)}</td>
                <td>{format!("{:.1}", s.std_dev)}</td>
            </tr>
        }
    };
    view! {
        <table class="stats">
            <thead>
                <tr>
                    <th></th>
                    <th scope="col">"min"</th>
                    <th scope="col">"25%"</th>
                    <th scope="col">"median"</th>
                    <th scope="col">"75%"</th>
                    <th scope="col">"max"</th>
                    <th scope="col">"mean"</th>
                    <th scope="col">"std dev"</th>
                </tr>
            </thead>
            <tbody>{row("return", r.returns, true)} {row("moves", r.lengths, false)}</tbody>
        </table>
    }
}

fn tiles(r: &BatchReport, ctx: AnalyseCtx) -> impl IntoView {
    let best = *r.best_episode();
    let tile = |label: &'static str, value: String, note: String| {
        view! {
            <div class="tile">
                <div class="tile-label">{label}</div>
                <div class="tile-value">{value}</div>
                <div class="tile-note">{note}</div>
            </div>
        }
    };
    view! {
        <div class="tiles">
            {tile("REACHED THE SHORE", percent(r.reached_goal_rate), format!("of {} voyages", r.episodes.len()))}
            {tile(
                "SHORTEST ROUTE",
                r.shortest_route_rate.map_or("—".into(), percent),
                r.best_possible.map_or(String::new(), |b| format!("return {}", signed(f64::from(b)))),
            )}
            {tile("MEAN RETURN", signed(f64::from(r.returns.mean)), format!("± {:.1} std dev", r.returns.std_dev))}
            {tile("MEDIAN RETURN", signed(f64::from(r.returns.median)), format!("worst {}", signed(f64::from(r.returns.min))))}
            <div class="tile best" class:hl-box=move || ctx.highlights.with(|h| h.contains(&UiElement::BestSeed))>
                <div class="tile-label">"BEST VOYAGE"</div>
                <div class="tile-value">{signed(f64::from(best.episode_return))}</div>
                <div class="tile-note">{format!("seed {} · {} moves", best.seed, best.length)}</div>
                <button class="primary" on:click=move |_| ctx.replay_best.run(best.seed)>"REPLAY BEST SEED ▶"</button>
            </div>
        </div>
    }
}

fn quiz(
    ctx: AnalyseCtx,
    level_id: String,
    task: usize,
    questions: Arc<Vec<QuestionDef>>,
) -> impl IntoView {
    questions
        .iter()
        .enumerate()
        .map(|(q, question)| {
            let level = level_id.clone();
            let done_memo = Memo::new(move |_| ctx.progress.with(|p| p.question_done(&level, task, q)));
            let done = move || done_memo.get();
            let outcome = move || ctx.feedback.with(|f| f.get(&(task, q)).copied());
            let options: Vec<String> = match question {
                QuestionDef::StatBand { bands, .. } => band_labels(bands),
                QuestionDef::Choice { options, .. } => options.clone(),
                QuestionDef::BestSeed { .. } => Vec::new(),
            };
            let explain = question.explain().to_string();
            let seed_input = RwSignal::new(String::new());
            let input = matches!(question, QuestionDef::BestSeed { .. }).then(|| {
                view! {
                    <div class="row">
                        "SEED"
                        <input
                            type="number"
                            min="0"
                            prop:value=move || seed_input.get()
                            on:input=move |e| seed_input.set(event_target_value(&e))
                            disabled=done
                        />
                        <button
                            disabled=done
                            on:click=move |_| {
                                if let Ok(seed) = seed_input.get_untracked().trim().parse::<u64>() {
                                    ctx.submit_answer.run((task, q, Answer::Seed(seed)));
                                }
                            }
                        >
                            "CHECK"
                        </button>
                    </div>
                }
            });
            view! {
                <div class="question" class:answered=done>
                    <p class="prompt">{format!("{}. {}", q + 1, question.prompt())}</p>
                    <div class="options">
                        {options
                            .into_iter()
                            .enumerate()
                            .map(|(i, label)| {
                                view! {
                                    <button
                                        disabled=done
                                        on:click=move |_| ctx.submit_answer.run((task, q, Answer::Option(i as u32)))
                                    >
                                        {label}
                                    </button>
                                }
                            })
                            .collect_view()}
                    </div>
                    {input}
                    <p class="feedback">
                        {move || {
                            if done() {
                                format!("✓ {explain}")
                            } else {
                                match outcome() {
                                    Some(AnswerOutcome::Incorrect) => "✗ Not quite. Check your statistics above.".into(),
                                    Some(AnswerOutcome::NeedsBatch) => "Run a batch first: this question is about your results.".into(),
                                    _ => String::new(),
                                }
                            }
                        }}
                    </p>
                </div>
            }
        })
        .collect_view()
}

pub fn analyse_view(ctx: AnalyseCtx) -> impl IntoView {
    let scatter_hover = RwSignal::new(None::<Hover>);
    let histogram_hover = RwSignal::new(None::<Hover>);
    let quiz_task = Memo::new(move |_| {
        ctx.snapshot.with(|s| {
            let s = s.as_ref()?;
            let GoalDef::Tasks(tasks) = &s.objective.as_ref()?.goal else {
                return None;
            };
            tasks.iter().enumerate().find_map(|(i, t)| match t {
                TaskDef::Quiz { questions } => {
                    Some((s.level_id.clone(), i, Arc::new(questions.clone())))
                }
                _ => None,
            })
        })
    });

    view! {
        <section class="analyse" class:hl-box=move || ctx.highlights.with(|h| h.contains(&UiElement::Analyse))>
            <h2>"EXPERIMENT · RANDOM AGENT"</h2>
            <p class="subtitle">
                "Run the agent on many seeds in a row. Every seed is a different voyage, and the same seed always repeats the same voyage."
            </p>
            <div class="row">
                "RUN"
                <select
                    prop:value=move || ctx.batch_size.get().to_string()
                    on:change=move |e| {
                        if let Ok(n) = event_target_value(&e).parse() {
                            ctx.batch_size.set(n)
                        }
                    }
                >
                    {BATCH_SIZES.map(|n| view! { <option value=n.to_string()>{n.to_string()}</option> })}
                </select>
                "VOYAGES FROM SEED"
                <input
                    type="number"
                    min="0"
                    prop:value=move || ctx.first_seed.get().to_string()
                    on:change=move |e| {
                        if let Ok(s) = event_target_value(&e).parse() {
                            ctx.first_seed.set(s)
                        }
                    }
                />
                <button class="primary" disabled=move || ctx.running.get() on:click=move |_| ctx.run_batch.run(())>
                    {move || if ctx.running.get() { "RUNNING…" } else { "▶ RUN BATCH" }}
                </button>
            </div>
            {move || {
                ctx.report
                    .with(|r| match r {
                        None => view! { <p class="placeholder">"No batch yet. Run one to see what the agent really achieves."</p> }.into_any(),
                        Some(r) => {
                            view! {
                                <p class="subtitle">
                                    {format!(
                                        "{} · seeds {} to {}",
                                        r.agent,
                                        r.first_seed,
                                        r.first_seed + r.episodes.len() as u64 - 1
                                    )}
                                </p>
                                {tiles(r, ctx)}
                                <h3>"RETURN OF EVERY VOYAGE"</h3>
                                {returns_chart(r, scatter_hover)}
                                <h3>"HOW THE RETURNS ARE SPREAD"</h3>
                                {histogram_chart(r, histogram_hover)}
                                <h3>"IN NUMBERS"</h3>
                                {stat_table(r)}
                            }
                                .into_any()
                        }
                    })
            }}
            {move || {
                quiz_task
                    .get()
                    .map(|(level, task, questions)| {
                        view! {
                            <div class="quiz" class:hl-box=move || ctx.highlights.with(|h| h.contains(&UiElement::Quiz))>
                                <h3>"CALYPSO'S QUESTIONS"</h3>
                                {quiz(ctx, level, task, questions)}
                            </div>
                        }
                    })
            }}
        </section>
    }
}
