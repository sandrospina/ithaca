//! The live Q-learning walkthrough panel, drawn under the arena: the
//! pseudocode executing with the raft's real numbers (docs/tutorials.md §3).

use leptos::prelude::*;
use rl_tutorial::{Depth, UiElement};

use crate::lesson_code::{self, ADVANCE, CHOOSE, Reveal, TARGET, UPDATE, cell, num};
use crate::protocol::LessonMode;
use crate::protocol::Snapshot;

/// Automatic stepping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Play {
    /// One move at a time at reading pace, stopping when the episode ends.
    Watch,
    /// A whole episode per step, episode after episode, to watch the policy form.
    Fast,
}

impl Play {
    pub fn delay_ms(self) -> u64 {
        match self {
            Play::Watch => 450,
            Play::Fast => 60,
        }
    }

    /// What each request asks for: a move to watch, or a whole episode.
    pub fn request(self) -> (LessonMode, Reveal) {
        match self {
            Play::Watch => (LessonMode::Move, Reveal::Move),
            Play::Fast => (LessonMode::Episode, Reveal::Episode),
        }
    }
}

#[derive(Clone, Copy)]
pub struct WalkCtx {
    pub snapshot: RwSignal<Option<Snapshot>>,
    pub depth: RwSignal<Depth>,
    pub line: RwSignal<usize>,
    pub play: RwSignal<Option<Play>>,
    pub highlights: Memo<Vec<UiElement>>,
    pub step_line: Callback<()>,
    pub step_move: Callback<()>,
    pub finish_episode: Callback<()>,
    pub set_play: Callback<Option<Play>>,
    pub back: Callback<()>,
}

const ACTIONS: [&str; 4] = ["N", "E", "S", "W"];

/// One Q-table row as a small table, with one action marked.
fn q_row(title: String, values: Vec<f32>, mark: Option<usize>, mark_note: &'static str) -> impl IntoView {
    view! {
        <div class="qrow">
            <div class="qrow-title">{title}</div>
            <div class="qrow-values">
                {values
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        view! {
                            <span class="qv" class:marked=mark == Some(i)>
                                <span class="qv-action">{ACTIONS.get(i).copied().unwrap_or("?")}</span>
                                <span class="qv-value">{num(*v)}</span>
                            </span>
                        }
                    })
                    .collect_view()}
            </div>
            <div class="qrow-note">{mark.map(|_| mark_note)}</div>
        </div>
    }
}

pub fn walkthrough_view(ctx: WalkCtx) -> impl IntoView {
    let hl = move |el: UiElement| ctx.highlights.with(|h| h.contains(&el));
    let header = move || {
        ctx.snapshot.with(|s| {
            let s = s.as_ref()?;
            let t = s.training.as_ref()?;
            let budget = t.budget.map_or(String::new(), |b| format!(" of {b}"));
            Some(match &s.lesson {
                Some(l) => {
                    let (moves, ret) = lesson_code::displayed_progress(l, ctx.line.get());
                    format!(
                        "Q-LEARNING · episode {} · move {moves} of {} · return {} · {} episodes trained{budget}",
                        l.episode,
                        l.max_steps,
                        num(ret),
                        t.episodes_trained
                    )
                }
                None => format!(
                    "Q-LEARNING · next: episode {} · {} episodes trained{budget}",
                    t.episodes_trained + 1,
                    t.episodes_trained
                ),
            })
        })
    };
    let rows = move || {
        let line = ctx.line.get();
        let depth = ctx.depth.get();
        ctx.snapshot.with(|s| {
            let s = s.as_ref();
            let lesson = s.and_then(|s| s.lesson.as_ref());
            let config = s.and_then(|s| s.training.as_ref()).map(|t| t.config).unwrap_or_default();
            let values = lesson.map(|l| lesson_code::line_values(l, line, &config)).unwrap_or_default();
            let current = lesson.map(|_| line);
            lesson_code::code_lines(depth)
                .iter()
                .enumerate()
                .map(|(i, code)| {
                    let is_current = current == Some(i);
                    view! {
                        <div class="code-line" class:current=is_current>
                            <span class="code-marker">{if is_current { "▶" } else { "" }}</span>
                            <span class="code-text">{*code}</span>
                            <span class="code-value">{values.get(i).cloned().unwrap_or_default()}</span>
                        </div>
                    }
                })
                .collect_view()
        })
    };
    let q_rows = move || {
        let line = ctx.line.get();
        ctx.snapshot.with(|s| {
            let m = s.as_ref()?.lesson.as_ref()?.last.clone()?;
            let t = &m.trace;
            let mut row = t.row_before.clone();
            if line >= UPDATE
                && let Some(v) = row.get_mut(t.action as usize)
            {
                *v = t.update.new_value;
            }
            let action = (line >= CHOOSE).then_some(t.action as usize);
            let note = if line >= UPDATE { "updated" } else { "chosen" };
            let s_row = q_row(format!("Q(s, ·) at {}", cell(m.from)), row, action, note);
            let next = (line >= TARGET).then(|| match t.next_max {
                Some(max) => {
                    let best = t.next_row.iter().position(|&v| (v - max).abs() < 1e-6);
                    q_row(format!("Q(s′, ·) at {}", cell(m.to)), t.next_row.clone(), best, "max").into_any()
                }
                None => view! {
                    <div class="qrow">
                        <div class="qrow-title">{format!("s′ = {} ends the voyage", cell(m.to))}</div>
                        <div class="qrow-note">"nothing follows, so nothing is added to r"</div>
                    </div>
                }
                .into_any(),
            });
            Some(view! { <div class="qrows">{s_row}{next}</div> })
        })
    };
    let finished = move || ctx.snapshot.with(|s| s.as_ref().and_then(|s| s.lesson.as_ref()).and_then(|l| l.finished));
    let budget_left = move || {
        ctx.snapshot.with(|s| {
            s.as_ref().and_then(|s| s.training.as_ref()).is_none_or(|t| t.budget.is_none_or(|b| t.episodes_trained < b))
        })
    };
    let under_way =
        move || ctx.snapshot.with(|s| s.as_ref().and_then(|s| s.lesson.as_ref()).is_some_and(|l| l.finished.is_none()));
    let can_step = move || under_way() || budget_left();
    // Whether this visit has stepped at all: the note on what these episodes are shows until then.
    let started = move || {
        ctx.line.get() > lesson_code::RESET
            || ctx.snapshot.with(|s| s.as_ref().and_then(|s| s.lesson.as_ref()).is_some_and(|l| l.moves > 0))
    };
    view! {
        <section class="walkthrough" class:hl-box=move || hl(UiElement::Walkthrough)>
            <div class="walk-head">
                <span class="glow">{header}</span>
                <span style="flex: 1"></span>
                <button class="quiet" on:click=move |_| ctx.back.run(())>"◀ LEARNING CURVE"</button>
            </div>
            <div class="code" role="log" aria-live="polite">{rows}</div>
            {q_rows}
            <div class="row actions">
                <button disabled=move || !can_step() on:click=move |_| ctx.step_line.run(())>
                    "STEP LINE (SPACE)"
                </button>
                <button disabled=move || !can_step() on:click=move |_| ctx.step_move.run(())>
                    "STEP MOVE (ENTER)"
                </button>
                <button disabled=move || !under_way() on:click=move |_| ctx.finish_episode.run(())>
                    "FINISH EPISODE"
                </button>
                {move || match ctx.play.get() {
                    Some(_) => view! { <button class="primary" on:click=move |_| ctx.set_play.run(None)>"■ PAUSE"</button> }.into_any(),
                    None => view! {
                        <button class="primary" disabled=move || !can_step() on:click=move |_| ctx.set_play.run(Some(Play::Watch))>
                            "▶ PLAY"
                        </button>
                        <button disabled=move || !can_step() on:click=move |_| ctx.set_play.run(Some(Play::Fast))>
                            "▶▶ FAST"
                        </button>
                    }
                    .into_any(),
                }}
            </div>
            {move || {
                let note = if !can_step() {
                    "The training budget is used up. Start a new raft in CONFIGURE to train again."
                } else if finished().is_some() && ctx.line.get() > ADVANCE {
                    "Episode over. It is now on the learning curve. Step again to begin the next one."
                } else if !started() {
                    "These are real training episodes of your raft: each one counts toward the budget, and bulk training carries on from here."
                } else {
                    return None;
                };
                Some(view! { <p class="subtitle">{note}</p> })
            }}
        </section>
    }
}
