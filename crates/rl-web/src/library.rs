//! The concept library: every RL concept met so far, at the three depths,
//! with further reading and a way back to the step that taught it.

use leptos::prelude::*;
use rl_tutorial::{ConceptLibrary, Depth};

use crate::campaign::Progress;
use crate::content::{self, LEVELS};

#[derive(Clone, Copy)]
pub struct LibraryCtx {
    pub progress: RwSignal<Progress>,
    pub depth: RwSignal<Depth>,
    pub unlock_all: bool,
    /// Opens a level at a tutorial step: (level index, step).
    pub replay: Callback<(usize, usize)>,
}

pub fn library_view(ctx: LibraryCtx) -> impl IntoView {
    let lib = StoredValue::new(ConceptLibrary::parse(content::CONCEPTS).ok());
    let open = RwSignal::new(None::<String>);
    let unlocked = move |c: &rl_tutorial::ConceptDef| {
        ctx.unlock_all || ctx.progress.with(|p| p.concept_unlocked(&c.level, c.step))
    };
    move || {
        let Some(lib) = lib.get_value() else {
            return view! { <p class="placeholder">"The concept library could not be read."</p> }.into_any();
        };
        let known = lib.concepts.iter().filter(|c| unlocked(c)).count();
        view! {
            <section class="training-view library">
                <h2>"LIBRARY · CONCEPTS"</h2>
                <p class="subtitle">
                    {format!(
                        "{known} of {} concepts met so far. Each one opens when the tutorial that teaches it reaches it.",
                        lib.concepts.len()
                    )}
                </p>
                {LEVELS
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| lib.concepts.iter().any(|c| c.level == l.id))
                    .map(|(li, l)| {
                        view! {
                            <h3>{format!("LEVEL {}", l.menu_label.to_uppercase())}</h3>
                            <ul class="concepts">
                                {lib.concepts
                                    .iter()
                                    .filter(|c| c.level == l.id)
                                    .map(|c| {
                                        if !unlocked(c) {
                                            return view! {
                                                <li class="concept locked">
                                                    <span class="concept-title">{format!("■ {}", c.title)}</span>
                                                    <span class="concept-gloss">"not met yet"</span>
                                                </li>
                                            }
                                            .into_any();
                                        }
                                        let c = c.clone();
                                        let id = c.id.clone();
                                        let is_open = {
                                            let id = id.clone();
                                            move || open.with(|o| o.as_deref() == Some(id.as_str()))
                                        };
                                        let step = c.step;
                                        view! {
                                            <li class="concept">
                                                <button
                                                    class="concept-head"
                                                    aria-expanded={
                                                        let is_open = is_open.clone();
                                                        move || is_open().to_string()
                                                    }
                                                    on:click={
                                                        let id = id.clone();
                                                        move |_| open.update(|o| *o = if o.as_deref() == Some(id.as_str()) { None } else { Some(id.clone()) })
                                                    }
                                                >
                                                    <span class="concept-title">{c.title.clone()}</span>
                                                    <span class="concept-gloss">{c.glossary.clone()}</span>
                                                </button>
                                                {move || {
                                                    is_open().then(|| {
                                                        let text = c.text.at(ctx.depth.get()).to_string();
                                                        view! {
                                                            <div class="concept-body">
                                                                <div class="depth" role="group" aria-label="Depth">
                                                                    {[(Depth::Story, "STORY"), (Depth::Intuition, "INTUITION"), (Depth::Math, "MATH")]
                                                                        .into_iter()
                                                                        .map(|(d, label)| view! {
                                                                            <button class="quiet" class:active=move || ctx.depth.get() == d on:click=move |_| ctx.depth.set(d)>{label}</button>
                                                                        })
                                                                        .collect_view()}
                                                                </div>
                                                                <p>{text}</p>
                                                                {(!c.reading.is_empty()).then(|| view! {
                                                                    <p class="reading">"Further reading: " {c.reading.join("; ")}</p>
                                                                })}
                                                                <button on:click=move |_| ctx.replay.run((li, step))>"REPLAY THE EXPLAINER ▶"</button>
                                                            </div>
                                                        }
                                                    })
                                                }}
                                            </li>
                                        }
                                        .into_any()
                                    })
                                    .collect_view()}
                            </ul>
                        }
                    })
                    .collect_view()}
            </section>
        }
        .into_any()
    }
}
