//! Phosphor Canvas2D renderer (docs/visual-style.md).
//!
//! Rules: colours come only from `rl_theme`; each RL concept has one glyph;
//! the renderer reads render models and never touches the simulation.

use std::f64::consts::{PI, TAU};

use rl_core::{RenderEvent, RenderModel};

use crate::protocol::CellPolicy;
use rl_theme as theme;
use web_sys::CanvasRenderingContext2d as Ctx;

/// Backing-store size of the arena canvas, in pixels.
pub const CANVAS_SIZE: f64 = 720.0;
const MARGIN: f64 = 36.0;
/// How long reward rings and bump sparks stay visible, in milliseconds.
const RING_MS: f64 = 450.0;
/// How long the phosphor trail behind a moving agent lasts, in milliseconds.
const TRAIL_MS: f64 = 700.0;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Effects {
    /// Plain mode: no glow, no persistence trails.
    pub plain: bool,
}

#[derive(Clone, Copy, Debug)]
struct Flash {
    position: (f64, f64),
    value: f64,
    born: f64,
    bump: bool,
}

/// Animation state kept between frames: smoothed agent positions and
/// short-lived flashes.
#[derive(Default)]
pub struct RenderState {
    displayed: Vec<(f64, f64)>,
    /// Recent displayed positions per agent, with timestamps: the phosphor trail.
    trails: Vec<Vec<(f64, f64, f64)>>,
    flashes: Vec<Flash>,
    last_frame: Option<f64>,
}

impl RenderState {
    /// Called once per new snapshot. `snap` jumps agents to their positions
    /// instead of gliding (used on reset).
    pub fn on_snapshot(&mut self, model: &RenderModel, now: f64, snap: bool) {
        if snap || self.displayed.len() != model.agents.len() {
            self.displayed = model.agents.iter().map(|a| (f64::from(a.position.0), f64::from(a.position.1))).collect();
            self.trails = vec![Vec::new(); model.agents.len()];
            self.flashes.clear();
        }
        for e in &model.events {
            match e {
                RenderEvent::Reward { position, value } => self.flashes.push(Flash {
                    position: (f64::from(position.0), f64::from(position.1)),
                    value: f64::from(*value),
                    born: now,
                    bump: false,
                }),
                RenderEvent::Bump { position } => self.flashes.push(Flash {
                    position: (f64::from(position.0), f64::from(position.1)),
                    value: 0.0,
                    born: now,
                    bump: true,
                }),
            }
        }
    }

    fn advance(&mut self, model: &RenderModel, now: f64) {
        let dt = self.last_frame.map_or(0.0, |t| (now - t).max(0.0));
        self.last_frame = Some(now);
        let k = (dt / 1000.0 * 18.0).min(1.0);
        for ((d, a), trail) in self.displayed.iter_mut().zip(&model.agents).zip(&mut self.trails) {
            d.0 += (f64::from(a.position.0) - d.0) * k;
            d.1 += (f64::from(a.position.1) - d.1) * k;
            let moved = trail.last().is_none_or(|&(x, y, _)| (x - d.0).abs() + (y - d.1).abs() > 0.01);
            if moved {
                trail.push((d.0, d.1, now));
            }
            trail.retain(|&(_, _, t)| now - t < TRAIL_MS);
        }
        self.flashes.retain(|f| now - f.born < RING_MS);
    }
}

/// How long the flash on an updated Q-value lasts, in milliseconds.
const UPDATE_MS: f64 = 1200.0;

/// One Q-value that just changed: its cell and action.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Updated {
    pub cell: (i32, i32),
    pub action: u32,
    pub born: f64,
}

/// Cell markers drawn over the map.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Marks {
    /// The cell whose Q-values the inspector shows.
    pub selected: Option<(i32, i32)>,
    pub updated: Option<Updated>,
}

/// Arena geometry: cell size and the top-left corner, in canvas pixels.
fn layout(model: &RenderModel) -> (f64, f64, f64) {
    let cols = f64::from(model.width);
    let rows = f64::from(model.height);
    let cell = ((CANVAS_SIZE - 2.0 * MARGIN) / cols).min((CANVAS_SIZE - 2.0 * MARGIN) / rows);
    (cell, (CANVAS_SIZE - cell * cols) / 2.0, (CANVAS_SIZE - cell * rows) / 2.0)
}

/// The cell under a point given in canvas pixels, if any.
pub fn cell_at(model: &RenderModel, x: f64, y: f64) -> Option<(i32, i32)> {
    let (cell, ox, oy) = layout(model);
    let (cx, cy) = (((x - ox) / cell).floor(), ((y - oy) / cell).floor());
    (cx >= 0.0 && cy >= 0.0 && cx < f64::from(model.width) && cy < f64::from(model.height))
        .then_some((cx as i32, cy as i32))
}

/// What is drawn over the world besides its own glyphs.
#[derive(Clone, Copy, Debug, Default)]
pub struct Layers<'a> {
    pub hl: Highlights,
    /// The learned Q-values, shown as shading and arrows.
    pub policy: Option<&'a [CellPolicy]>,
    pub marks: Marks,
}

/// Which elements a tutorial step is pointing at.
#[derive(Clone, Copy, Debug, Default)]
pub struct Highlights {
    pub agent: bool,
    pub goal: bool,
}

fn set_glow(ctx: &Ctx, colour: &str, blur: f64, fx: Effects) {
    if fx.plain {
        ctx.set_shadow_blur(0.0);
    } else {
        ctx.set_shadow_color(colour);
        ctx.set_shadow_blur(blur);
    }
}

fn clear(ctx: &Ctx) {
    ctx.set_shadow_blur(0.0);
    ctx.set_global_alpha(1.0);
    ctx.set_fill_style_str(theme::ARENA_BG.hex);
    ctx.fill_rect(0.0, 0.0, CANVAS_SIZE, CANVAS_SIZE);
}

/// Phosphor persistence: a fading line behind each moving agent. Drawn
/// explicitly rather than by translucent clearing, which leaves ghosts.
fn draw_trail(
    ctx: &Ctx,
    trail: &[(f64, f64, f64)],
    colour: &str,
    now: f64,
    to_px: &dyn Fn(f64, f64) -> (f64, f64),
    cell: f64,
    fx: Effects,
) {
    if fx.plain {
        return;
    }
    set_glow(ctx, colour, 8.0, fx);
    ctx.set_stroke_style_str(colour);
    ctx.set_line_cap("round");
    for w in trail.windows(2) {
        let age = (now - w[1].2) / TRAIL_MS;
        ctx.set_global_alpha((0.55 * (1.0 - age)).clamp(0.0, 1.0));
        ctx.set_line_width(cell * 0.06 * (1.0 - age).max(0.2));
        let (x0, y0) = to_px(w[0].0, w[0].1);
        let (x1, y1) = to_px(w[1].0, w[1].1);
        ctx.begin_path();
        ctx.move_to(x0, y0);
        ctx.line_to(x1, y1);
        ctx.stroke();
    }
    ctx.set_global_alpha(1.0);
}

/// Draws one frame of a world.
pub fn draw_world(ctx: &Ctx, model: &RenderModel, state: &mut RenderState, now: f64, fx: Effects, layers: Layers<'_>) {
    let Layers { hl, policy, marks } = layers;
    state.advance(model, now);
    clear(ctx);

    let cols = f64::from(model.width);
    let rows = f64::from(model.height);
    let (cell, ox, oy) = layout(model);
    let to_px = |x: f64, y: f64| (ox + x * cell, oy + y * cell);

    // Graticule.
    ctx.set_shadow_blur(0.0);
    ctx.set_line_width(1.0);
    ctx.set_stroke_style_str(&theme::rgba(theme::TERRAIN.hex, 0.13));
    ctx.begin_path();
    for i in 0..=model.width as i32 {
        let (x, _) = to_px(f64::from(i), 0.0);
        ctx.move_to(x, oy);
        ctx.line_to(x, oy + rows * cell);
    }
    for j in 0..=model.height as i32 {
        let (_, y) = to_px(0.0, f64::from(j));
        ctx.move_to(ox, y);
        ctx.line_to(ox + cols * cell, y);
    }
    ctx.stroke();

    // Border.
    set_glow(ctx, theme::TERRAIN.hex, 8.0, fx);
    ctx.set_line_width(2.0);
    ctx.set_stroke_style_str(theme::TERRAIN.hex);
    ctx.stroke_rect(ox, oy, cols * cell, rows * cell);

    if let Some(policy) = policy {
        draw_policy_overlay(ctx, policy, &to_px, cell, fx);
    }
    if let Some(u) = marks.updated {
        let t = (now - u.born) / UPDATE_MS;
        if (0.0..1.0).contains(&t) {
            let (px, py) = to_px(f64::from(u.cell.0), f64::from(u.cell.1));
            draw_updated_value(ctx, px, py, cell, u.action, t, fx);
        }
    }
    for &(x, y) in &model.walls {
        let (px, py) = to_px(f64::from(x), f64::from(y));
        draw_wall(ctx, px, py, cell, fx);
    }
    for &(x, y) in &model.wind {
        let (px, py) = to_px(f64::from(x), f64::from(y));
        draw_wind(ctx, px, py, cell, now, fx);
    }
    for &(x, y) in &model.hazards {
        let (px, py) = to_px(f64::from(x), f64::from(y));
        draw_hazard(ctx, px, py, cell, false, fx);
    }
    for &(x, y) in &model.lethal {
        let (px, py) = to_px(f64::from(x), f64::from(y));
        draw_hazard(ctx, px, py, cell, true, fx);
    }
    for &(x, y) in &model.lures {
        let (px, py) = to_px(f64::from(x) + 0.5, f64::from(y) + 0.5);
        draw_lotus(ctx, px, py, cell, now, fx);
    }
    for &(x, y) in &model.goals {
        let (px, py) = to_px(f64::from(x) + 0.5, f64::from(y) + 0.5);
        draw_goal(ctx, px, py, cell, now, fx);
        if hl.goal {
            draw_brackets(ctx, px, py, cell, now, fx);
        }
    }

    for f in &state.flashes {
        let (px, py) = to_px(f.position.0, f.position.1);
        let t = (now - f.born) / RING_MS;
        if f.bump {
            draw_spark(ctx, px, py, cell, t, fx);
        } else {
            draw_reward_ring(ctx, px, py, cell, t, f.value, fx);
        }
    }

    for (a, trail) in model.agents.iter().zip(&state.trails) {
        draw_trail(ctx, trail, theme::glyph_colour(a.glyph), now, &to_px, cell, fx);
    }
    for (a, &(dx, dy)) in model.agents.iter().zip(&state.displayed) {
        let (px, py) = to_px(dx, dy);
        match a.glyph {
            rl_core::Glyph::Adversary => draw_adversary(ctx, px, py, cell, fx),
            _ => draw_agent(ctx, px, py, cell, f64::from(a.heading), fx),
        }
        if hl.agent {
            draw_brackets(ctx, px, py, cell, now, fx);
        }
    }
    if let Some((x, y)) = marks.selected {
        let (px, py) = to_px(f64::from(x) + 0.5, f64::from(y) + 0.5);
        draw_brackets(ctx, px, py, cell, now, fx);
    }
    ctx.set_shadow_blur(0.0);
}

/// A Q-value just changed: the cell flashes and the edge on the side of
/// the updated action lights up, fading over `UPDATE_MS`.
fn draw_updated_value(ctx: &Ctx, x: f64, y: f64, cell: f64, action: u32, t: f64, fx: Effects) {
    let fade = 1.0 - t;
    ctx.set_shadow_blur(0.0);
    ctx.set_fill_style_str(&theme::rgba(theme::POLICY.hex, 0.35 * fade));
    ctx.fill_rect(x + 1.0, y + 1.0, cell - 2.0, cell - 2.0);
    let inset = cell * 0.08;
    let (x0, y0, x1, y1) = match action {
        0 => (x + inset, y + 3.0, x + cell - inset, y + 3.0),
        1 => (x + cell - 3.0, y + inset, x + cell - 3.0, y + cell - inset),
        2 => (x + inset, y + cell - 3.0, x + cell - inset, y + cell - 3.0),
        _ => (x + 3.0, y + inset, x + 3.0, y + cell - inset),
    };
    set_glow(ctx, theme::POLICY.hex, 10.0, fx);
    ctx.set_global_alpha(fade);
    ctx.set_line_width(4.0);
    ctx.set_stroke_style_str(theme::POLICY.hex);
    ctx.begin_path();
    ctx.move_to(x0, y0);
    ctx.line_to(x1, y1);
    ctx.stroke();
    ctx.set_global_alpha(1.0);
}

/// The learned policy: each cell shaded by its best Q-value (green positive,
/// magenta negative, normalised across the map) and an arrow for the
/// preferred move, longer when the preference is clearer.
fn draw_policy_overlay(
    ctx: &Ctx,
    policy: &[CellPolicy],
    to_px: &dyn Fn(f64, f64) -> (f64, f64),
    cell: f64,
    fx: Effects,
) {
    let best = |v: &[f32; 4]| v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let scale = policy.iter().map(|c| best(&c.values).abs()).fold(0.0f32, f32::max);
    if scale <= 1e-6 {
        return; // untrained: nothing learned yet
    }
    for c in policy {
        let (x, y) = (f64::from(c.cell.0), f64::from(c.cell.1));
        let (px, py) = to_px(x, y);
        let top = best(&c.values);
        draw_value_cell(ctx, px, py, cell, f64::from(top / scale));
        let low = c.values.iter().copied().fold(f32::INFINITY, f32::min);
        if top - low < 1e-6 {
            continue; // no preference yet
        }
        let action = c.values.iter().position(|&v| v == top).unwrap_or(0);
        let second = c.values.iter().copied().filter(|&v| v < top).fold(f32::NEG_INFINITY, f32::max).max(low);
        let strength = f64::from(((top - second) / (top - low)).clamp(0.15, 1.0));
        let heading = [-PI / 2.0, 0.0, PI / 2.0, PI][action];
        let (cx, cy) = to_px(x + 0.5, y + 0.5);
        draw_policy_arrow(ctx, cx, cy, cell, heading, strength, fx);
    }
}

// ---------------------------------------------------------------- glyphs

/// Wind: faint streamlines drifting across the cell, in terrain green. Shape
/// alone tells it apart; it sits under everything else.
pub fn draw_wind(ctx: &Ctx, x: f64, y: f64, cell: f64, now: f64, fx: Effects) {
    set_glow(ctx, theme::TERRAIN.hex, 3.0, fx);
    ctx.set_line_width(1.2);
    ctx.set_stroke_style_str(&theme::rgba(theme::TERRAIN.hex, 0.38));
    let drift = (now / 1100.0 + x * 0.013).sin() * cell * 0.06;
    for k in 0..3 {
        let yy = y + cell * (0.3 + 0.2 * f64::from(k));
        let x0 = x + cell * 0.18 + drift * if k == 1 { -1.0 } else { 1.0 };
        let w = cell * 0.64;
        ctx.begin_path();
        ctx.move_to(x0, yy);
        ctx.bezier_curve_to(x0 + w * 0.25, yy - cell * 0.06, x0 + w * 0.5, yy + cell * 0.06, x0 + w * 0.75, yy);
        ctx.line_to(x0 + w, yy - cell * 0.03);
        ctx.stroke();
    }
}

/// Lure: a lotus flower outline in gold. Tempting like the goal, told apart by shape.
pub fn draw_lotus(ctx: &Ctx, cx: f64, cy: f64, cell: f64, now: f64, fx: Effects) {
    let r = cell * 0.26;
    let sway = (now / 1400.0).sin() * 0.06;
    set_glow(ctx, theme::GOAL.hex, 8.0, fx);
    ctx.set_line_width(1.5);
    ctx.set_stroke_style_str(theme::GOAL.hex);
    ctx.set_global_alpha(0.85);
    for k in 0..5 {
        let a = -PI / 2.0 + (f64::from(k) - 2.0) * 0.55 + sway;
        let (tx, ty) = (cx + a.cos() * r, cy + 0.25 * r + a.sin() * r);
        let base = (cx, cy + 0.35 * r);
        let side = 0.22;
        ctx.begin_path();
        ctx.move_to(base.0, base.1);
        ctx.quadratic_curve_to(base.0 + (a - side).cos() * r * 0.9, base.1 + (a - side).sin() * r * 0.9, tx, ty);
        ctx.quadratic_curve_to(
            base.0 + (a + side).cos() * r * 0.9,
            base.1 + (a + side).sin() * r * 0.9,
            base.0,
            base.1,
        );
        ctx.stroke();
    }
    ctx.begin_path();
    ctx.move_to(cx - r * 0.9, cy + r * 0.55);
    ctx.line_to(cx + r * 0.9, cy + r * 0.55);
    ctx.stroke();
    ctx.set_global_alpha(1.0);
}

/// Player agent: a triangle pointing along its heading.
pub fn draw_agent(ctx: &Ctx, cx: f64, cy: f64, cell: f64, heading: f64, fx: Effects) {
    let r = cell * 0.32;
    let pts = [(r, 0.0), (-r * 0.75, r * 0.7), (-r * 0.45, 0.0), (-r * 0.75, -r * 0.7)];
    let (s, c) = heading.sin_cos();
    set_glow(ctx, theme::AGENT.hex, 14.0, fx);
    ctx.set_line_width(2.5);
    ctx.set_stroke_style_str(theme::AGENT.hex);
    ctx.set_fill_style_str(&theme::rgba(theme::AGENT.hex, 0.18));
    ctx.begin_path();
    for (i, (x, y)) in pts.iter().enumerate() {
        let (px, py) = (cx + x * c - y * s, cy + x * s + y * c);
        if i == 0 {
            ctx.move_to(px, py);
        } else {
            ctx.line_to(px, py);
        }
    }
    ctx.close_path();
    ctx.fill();
    ctx.stroke();
}

/// Adversary: a jagged diamond.
pub fn draw_adversary(ctx: &Ctx, cx: f64, cy: f64, cell: f64, fx: Effects) {
    let r = cell * 0.34;
    set_glow(ctx, theme::ADVERSARY.hex, 14.0, fx);
    ctx.set_line_width(2.5);
    ctx.set_stroke_style_str(theme::ADVERSARY.hex);
    ctx.begin_path();
    let n = 8;
    for i in 0..=n {
        let a = f64::from(i) / f64::from(n) * TAU - PI / 2.0;
        let rr = if i % 2 == 0 { r } else { r * 0.55 };
        let (px, py) = (cx + rr * a.cos(), cy + rr * a.sin());
        if i == 0 {
            ctx.move_to(px, py);
        } else {
            ctx.line_to(px, py);
        }
    }
    ctx.stroke();
}

/// Goal: a circle with radiating ticks that slowly turn.
pub fn draw_goal(ctx: &Ctx, cx: f64, cy: f64, cell: f64, now: f64, fx: Effects) {
    let r = cell * 0.2;
    set_glow(ctx, theme::GOAL.hex, 12.0, fx);
    ctx.set_line_width(2.0);
    ctx.set_stroke_style_str(theme::GOAL.hex);
    ctx.begin_path();
    let _ = ctx.arc(cx, cy, r, 0.0, TAU);
    ctx.stroke();
    let spin = now / 4000.0;
    ctx.begin_path();
    for i in 0..8 {
        let a = f64::from(i) / 8.0 * TAU + spin;
        ctx.move_to(cx + a.cos() * r * 1.45, cy + a.sin() * r * 1.45);
        ctx.line_to(cx + a.cos() * r * 1.95, cy + a.sin() * r * 1.95);
    }
    ctx.stroke();
}

/// Wall: a solid block outline with an inset, in terrain green.
pub fn draw_wall(ctx: &Ctx, x: f64, y: f64, cell: f64, fx: Effects) {
    set_glow(ctx, theme::TERRAIN.hex, 8.0, fx);
    ctx.set_line_width(2.0);
    ctx.set_stroke_style_str(theme::TERRAIN.hex);
    ctx.stroke_rect(x + 3.0, y + 3.0, cell - 6.0, cell - 6.0);
    ctx.set_line_width(1.0);
    ctx.stroke_rect(x + cell * 0.25, y + cell * 0.25, cell * 0.5, cell * 0.5);
}

/// Hazard: cross-hatched cell; lethal hazards are double-hatched.
pub fn draw_hazard(ctx: &Ctx, x: f64, y: f64, cell: f64, lethal: bool, fx: Effects) {
    set_glow(ctx, theme::HAZARD.hex, 6.0, fx);
    ctx.set_line_width(1.5);
    ctx.set_stroke_style_str(theme::HAZARD.hex);
    ctx.stroke_rect(x + 3.0, y + 3.0, cell - 6.0, cell - 6.0);
    ctx.save();
    ctx.begin_path();
    ctx.rect(x + 3.0, y + 3.0, cell - 6.0, cell - 6.0);
    ctx.clip();
    ctx.begin_path();
    let step = cell / 6.0;
    let mut d = -cell;
    while d < cell {
        ctx.move_to(x + d, y);
        ctx.line_to(x + d + cell, y + cell);
        if lethal {
            ctx.move_to(x + d + cell, y);
            ctx.line_to(x + d, y + cell);
        }
        d += step;
    }
    ctx.stroke();
    ctx.restore();
}

/// Reward event: an expanding ring, gold for positive, magenta for negative.
pub fn draw_reward_ring(ctx: &Ctx, cx: f64, cy: f64, cell: f64, t: f64, value: f64, fx: Effects) {
    let colour = if value >= 0.0 { theme::GOAL.hex } else { theme::ADVERSARY.hex };
    set_glow(ctx, colour, 12.0, fx);
    ctx.set_global_alpha((1.0 - t).clamp(0.0, 1.0));
    ctx.set_line_width(2.0);
    ctx.set_stroke_style_str(colour);
    ctx.begin_path();
    let _ = ctx.arc(cx, cy, cell * (0.2 + 0.6 * t), 0.0, TAU);
    ctx.stroke();
    ctx.set_global_alpha(1.0);
}

/// Bump: a short spark where the move was blocked.
fn draw_spark(ctx: &Ctx, cx: f64, cy: f64, cell: f64, t: f64, fx: Effects) {
    set_glow(ctx, theme::TERRAIN.hex, 10.0, fx);
    ctx.set_global_alpha((1.0 - t).clamp(0.0, 1.0));
    ctx.set_line_width(2.0);
    ctx.set_stroke_style_str(theme::TERRAIN.hex);
    ctx.begin_path();
    let r = cell * 0.15 * (1.0 + t);
    for i in 0..4 {
        let a = f64::from(i) * PI / 2.0 + PI / 4.0;
        ctx.move_to(cx + a.cos() * r * 0.4, cy + a.sin() * r * 0.4);
        ctx.line_to(cx + a.cos() * r, cy + a.sin() * r);
    }
    ctx.stroke();
    ctx.set_global_alpha(1.0);
}

/// Policy: a thin arrow; length and brightness follow `strength` in [0, 1].
pub fn draw_policy_arrow(ctx: &Ctx, cx: f64, cy: f64, cell: f64, heading: f64, strength: f64, fx: Effects) {
    let len = cell * (0.15 + 0.25 * strength);
    let (s, c) = heading.sin_cos();
    set_glow(ctx, theme::POLICY.hex, 6.0, fx);
    ctx.set_global_alpha(0.35 + 0.65 * strength);
    ctx.set_line_width(1.5);
    ctx.set_stroke_style_str(theme::POLICY.hex);
    ctx.begin_path();
    let (tx, ty) = (cx + c * len, cy + s * len);
    ctx.move_to(cx - c * len, cy - s * len);
    ctx.line_to(tx, ty);
    let head = cell * 0.09;
    for side in [-1.0, 1.0] {
        let a = heading + PI - side * 0.5;
        ctx.move_to(tx, ty);
        ctx.line_to(tx + a.cos() * head, ty + a.sin() * head);
    }
    ctx.stroke();
    ctx.set_global_alpha(1.0);
}

/// Value cell: brightness follows |value|; green positive, magenta negative.
pub fn draw_value_cell(ctx: &Ctx, x: f64, y: f64, cell: f64, value: f64) {
    let colour = if value >= 0.0 { theme::TERRAIN.hex } else { theme::ADVERSARY.hex };
    ctx.set_shadow_blur(0.0);
    ctx.set_fill_style_str(&theme::rgba(colour, 0.08 + 0.5 * value.abs().min(1.0)));
    ctx.fill_rect(x + 1.0, y + 1.0, cell - 2.0, cell - 2.0);
}

/// Fog: dim, noisy, scanlined. Debug view of unobserved true state.
pub fn draw_fog(ctx: &Ctx, x: f64, y: f64, cell: f64, now: f64) {
    ctx.set_shadow_blur(0.0);
    ctx.set_fill_style_str(&theme::rgba(theme::TERRAIN.hex, 0.06));
    ctx.fill_rect(x, y, cell, cell);
    let mut seed = (now / 90.0) as u32;
    ctx.set_fill_style_str(&theme::rgba(theme::TERRAIN.hex, 0.25));
    for _ in 0..18 {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let px = f64::from(seed % 997) / 997.0 * cell;
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let py = f64::from(seed % 991) / 991.0 * cell;
        ctx.fill_rect(x + px, y + py, 1.5, 1.5);
    }
    ctx.set_fill_style_str(&theme::rgba(theme::ARENA_BG.hex, 0.5));
    let mut yy = y;
    while yy < y + cell {
        ctx.fill_rect(x, yy, cell, 1.0);
        yy += 3.0;
    }
}

/// Message glyph (Level 15): a Greek letter in the sender's colour.
pub fn draw_message(ctx: &Ctx, cx: f64, cy: f64, cell: f64, symbol: &str, colour: &str, fx: Effects) {
    set_glow(ctx, colour, 10.0, fx);
    ctx.set_fill_style_str(colour);
    ctx.set_font(&format!("{}px 'IBM Plex Mono', monospace", (cell * 0.45) as u32));
    ctx.set_text_align("center");
    ctx.set_text_baseline("middle");
    let _ = ctx.fill_text(symbol, cx, cy);
}

/// Resource bar: segmented, always drawn in the same place on screen.
/// `rect` is `(x, y, width, height)` in pixels.
pub fn draw_resource_bar(ctx: &Ctx, rect: (f64, f64, f64, f64), fraction: f64, colour: &str, fx: Effects) {
    let (x, y, w, h) = rect;
    let segments = 10;
    let gap = 2.0;
    let seg_w = (w - gap * f64::from(segments - 1)) / f64::from(segments);
    let lit = (fraction * f64::from(segments)).round() as i32;
    set_glow(ctx, colour, 6.0, fx);
    for i in 0..segments {
        let sx = x + f64::from(i) * (seg_w + gap);
        if i < lit {
            ctx.set_fill_style_str(colour);
            ctx.fill_rect(sx, y, seg_w, h);
        } else {
            ctx.set_stroke_style_str(&theme::rgba(colour, 0.35));
            ctx.set_line_width(1.0);
            ctx.stroke_rect(sx + 0.5, y + 0.5, seg_w - 1.0, h - 1.0);
        }
    }
}

/// Tutorial highlight: pulsing corner brackets in highlighted amber.
fn draw_brackets(ctx: &Ctx, cx: f64, cy: f64, cell: f64, now: f64, fx: Effects) {
    let pulse = 0.5 + 0.5 * (now / 300.0).sin();
    let r = cell * (0.48 + 0.04 * pulse);
    let l = cell * 0.16;
    set_glow(ctx, theme::SHELL_HI.hex, 8.0, fx);
    ctx.set_line_width(2.0);
    ctx.set_stroke_style_str(theme::SHELL_HI.hex);
    ctx.begin_path();
    for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        let (x, y) = (cx + sx * r, cy + sy * r);
        ctx.move_to(x, y - sy * l);
        ctx.line_to(x, y);
        ctx.line_to(x - sx * l, y);
    }
    ctx.stroke();
}

// ---------------------------------------------------------- style sheet

/// The style-sheet scene: every glyph and token side by side. It is the
/// visual reference for new levels (docs/visual-style.md §6).
pub fn draw_style_sheet(ctx: &Ctx, now: f64, fx: Effects) {
    clear(ctx);
    const COLS: usize = 5;
    let tile_w = CANVAS_SIZE / COLS as f64;
    let tile_h = CANVAS_SIZE / 4.0;
    let cell = tile_w.min(tile_h) * 0.5;
    let label = |i: usize, text: &str, token: &str| {
        let (col, row) = ((i % COLS) as f64, (i / COLS) as f64);
        ctx.set_shadow_blur(0.0);
        ctx.set_fill_style_str(theme::SHELL_FG.hex);
        ctx.set_font("13px 'IBM Plex Mono', monospace");
        ctx.set_text_align("center");
        ctx.set_text_baseline("alphabetic");
        let _ = ctx.fill_text(text, col * tile_w + tile_w / 2.0, row * tile_h + tile_h - 30.0);
        ctx.set_fill_style_str(theme::SHELL_DIM.hex);
        ctx.set_font("11px 'IBM Plex Mono', monospace");
        let _ = ctx.fill_text(token, col * tile_w + tile_w / 2.0, row * tile_h + tile_h - 14.0);
    };
    let centre = |i: usize| {
        let (col, row) = ((i % COLS) as f64, (i / COLS) as f64);
        (col * tile_w + tile_w / 2.0, row * tile_h + tile_h / 2.0 - 18.0)
    };

    let items: [(&str, &str); 18] = [
        ("agent", theme::AGENT.name),
        ("adversary", theme::ADVERSARY.name),
        ("goal", theme::GOAL.name),
        ("wall", theme::TERRAIN.name),
        ("hazard", theme::HAZARD.name),
        ("lethal hazard", theme::HAZARD.name),
        ("reward +", theme::GOAL.name),
        ("reward −", theme::ADVERSARY.name),
        ("policy arrows", theme::POLICY.name),
        ("value + / −", "terrain / adversary"),
        ("fog (unobserved)", "terrain 15% + noise"),
        ("message Δ", "sender colour"),
        ("resources", "per resource"),
        ("highlight", theme::SHELL_HI.name),
        ("teammate", theme::AGENT.name),
        ("bump", theme::TERRAIN.name),
        ("lure (lotus)", theme::GOAL.name),
        ("wind", "terrain 38%"),
    ];
    for (i, (text, token)) in items.iter().enumerate() {
        let (cx, cy) = centre(i);
        let (x0, y0) = (cx - cell / 2.0, cy - cell / 2.0);
        let t = (now % 900.0) / 900.0;
        match i {
            0 => draw_agent(ctx, cx, cy, cell, now / 2500.0, fx),
            1 => draw_adversary(ctx, cx, cy, cell, fx),
            2 => draw_goal(ctx, cx, cy, cell, now, fx),
            3 => draw_wall(ctx, x0, y0, cell, fx),
            4 => draw_hazard(ctx, x0, y0, cell, false, fx),
            5 => draw_hazard(ctx, x0, y0, cell, true, fx),
            6 => draw_reward_ring(ctx, cx, cy, cell, t, 1.0, fx),
            7 => draw_reward_ring(ctx, cx, cy, cell, t, -1.0, fx),
            8 => {
                let h = cell / 2.0;
                draw_policy_arrow(ctx, cx - h / 2.0, cy - h / 2.0, h, 0.0, 1.0, fx);
                draw_policy_arrow(ctx, cx + h / 2.0, cy - h / 2.0, h, PI / 2.0, 0.7, fx);
                draw_policy_arrow(ctx, cx - h / 2.0, cy + h / 2.0, h, PI, 0.4, fx);
                draw_policy_arrow(ctx, cx + h / 2.0, cy + h / 2.0, h, -PI / 2.0, 0.1, fx);
            }
            9 => {
                let h = cell / 2.0;
                for (k, v) in [1.0, 0.4, -0.3, -0.9].iter().enumerate() {
                    draw_value_cell(ctx, x0 + (k % 2) as f64 * h, y0 + (k / 2) as f64 * h, h, *v);
                }
            }
            10 => draw_fog(ctx, x0, y0, cell, now),
            11 => draw_message(ctx, cx, cy, cell, "Δ", theme::AGENT.hex, fx),
            12 => {
                draw_resource_bar(ctx, (x0 - 10.0, cy - 16.0, cell + 20.0, 10.0), 0.7, theme::GOAL.hex, fx);
                draw_resource_bar(ctx, (x0 - 10.0, cy + 4.0, cell + 20.0, 10.0), 0.3, theme::HAZARD.hex, fx);
            }
            13 => draw_brackets(ctx, cx, cy, cell, now, fx),
            14 => {
                ctx.set_global_alpha(0.6);
                draw_agent(ctx, cx, cy, cell, -PI / 2.0, fx);
                ctx.set_global_alpha(1.0);
            }
            15 => draw_spark(ctx, cx, cy, cell, t, fx),
            16 => draw_lotus(ctx, cx, cy, cell, now, fx),
            _ => draw_wind(ctx, x0, y0, cell, now, fx),
        }
        label(i, text, token);
    }
    ctx.set_shadow_blur(0.0);
}
