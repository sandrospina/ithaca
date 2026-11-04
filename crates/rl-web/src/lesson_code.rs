//! The live Q-learning walkthrough's logic: the pseudocode, the values each
//! line shows, and how stepping moves through it. Plain Rust, so it is tested
//! natively; `walkthrough` draws it.
//!
//! The numbers all come from the real training step ([`TransitionTrace`]):
//! stepping line by line only reveals them one at a time.

use rl_core::RenderModel;
use rl_tutorial::Depth;

use crate::protocol::{CellPolicy, EpisodeOutcome, LessonMode, LessonState, QLearningConfig};

pub const LINE_COUNT: usize = 9;
pub const RESET: usize = 1;
pub const CHOOSE: usize = 3;
pub const ACT: usize = 4;
pub const TARGET: usize = 5;
pub const UPDATE: usize = 6;
pub const ADVANCE: usize = 7;
pub const UNTIL: usize = 8;

const CODE: [&str; LINE_COUNT] = [
    "for each episode:",
    "  s ← reset(seed)",
    "  repeat:",
    "    a ← ε-greedy(Q, s)",
    "    take a, observe r, s′",
    "    target ← r + γ · max_a′ Q(s′, a′)",
    "    Q(s, a) ← Q(s, a) + α · (target − Q(s, a))",
    "    s ← s′",
    "  until s′ ends the voyage or time runs out",
];

const STORY: [&str; LINE_COUNT] = [
    "every voyage:",
    "  start from the spawn",
    "  every stroke:",
    "    pick the stroke that looks best, now and then a random one",
    "    make it, and see what it earns and where it lands",
    "    guess its worth: what it earned, plus part of the best to come",
    "    nudge the memory a little toward that guess",
    "    carry on from the new spot",
    "  until home, lost, or out of time",
];

/// The pseudocode, in plain words at story depth.
pub fn code_lines(depth: Depth) -> &'static [&'static str; LINE_COUNT] {
    match depth {
        Depth::Story => &STORY,
        Depth::Intuition | Depth::Math => &CODE,
    }
}

/// A number as the walkthrough shows it: two decimals, a real minus sign.
pub fn num(v: f32) -> String {
    let v = if v.abs() < 0.005 { 0.0 } else { v };
    format!("{v:.2}").replace('-', "−")
}

/// A number used after an operator: negative values get parentheses.
fn operand(v: f32) -> String {
    let n = num(v);
    if n.starts_with('−') { format!("({n})") } else { n }
}

pub fn cell(c: (i32, i32)) -> String {
    format!("({}, {})", c.0, c.1)
}

/// The live values beside each line, up to and including `line`. Lines not
/// reached yet are empty.
pub fn line_values(l: &LessonState, line: usize, config: &QLearningConfig) -> [String; LINE_COUNT] {
    let mut v: [String; LINE_COUNT] = Default::default();
    v[RESET] = match &l.last {
        None => format!("seed {} · s = {}", l.seed, cell(l.cell)),
        Some(_) => format!("seed {}", l.seed),
    };
    let Some(m) = &l.last else { return v };
    let t = &m.trace;
    let shown = |i: usize| line >= i;
    if shown(CHOOSE) {
        let best = t.row_before.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let ties = t.row_before.iter().filter(|&&q| q >= best - 1e-6).count();
        let how = if t.explored {
            "random move"
        } else if ties > 1 {
            "best move, a tie picked at random"
        } else {
            "best move"
        };
        v[CHOOSE] = format!("ε = {} · {how} → {}", num(config.epsilon), m.action_label.to_uppercase());
    }
    if shown(ACT) {
        let bump = if m.from == m.to && !t.terminated { " · bumped the edge" } else { "" };
        v[ACT] = format!("r = {} · s′ = {}{bump}", num(t.reward), cell(m.to));
    }
    if shown(TARGET) {
        v[TARGET] = match t.next_max {
            Some(max) => {
                let note = if t.truncated { " · out of time, s′ still counts" } else { "" };
                format!("{} + {} · {} = {}{note}", num(t.reward), num(config.gamma), operand(max), num(t.update.target))
            }
            None => format!("s′ ends the voyage: target = r = {}", num(t.update.target)),
        };
    }
    if shown(UPDATE) {
        let u = &t.update;
        v[UPDATE] = format!(
            "{} + {} · ({} − {}) = {}",
            num(u.old_value),
            num(config.alpha),
            num(u.target),
            operand(u.old_value),
            num(u.new_value)
        );
    }
    if shown(ADVANCE) {
        v[ADVANCE] = format!("s = {}", cell(m.to));
    }
    v[UNTIL] = match l.finished {
        Some(done) if shown(UNTIL) => {
            let how = match done.outcome {
                EpisodeOutcome::ReachedGoal => "reached the shore",
                EpisodeOutcome::Died => "swallowed by a whirlpool",
                EpisodeOutcome::OutOfTime => "out of time",
            };
            format!("{how} · return {} in {} moves", num(done.episode_return), done.length)
        }
        _ => format!("move {} of {}", t.step, l.max_steps),
    };
    v
}

/// Moves made and return so far as of `line`: before the move is taken, the
/// move just computed does not count yet.
pub fn displayed_progress(l: &LessonState, line: usize) -> (u32, f32) {
    match &l.last {
        Some(m) if line < ACT => (l.moves.saturating_sub(1), l.episode_return - m.trace.reward),
        _ => (l.moves, l.episode_return),
    }
}

/// The state s as of `line`: it becomes s′ on the line `s ← s′`.
pub fn displayed_state(l: &LessonState, line: usize) -> (u32, (i32, i32)) {
    match &l.last {
        Some(m) if line < ADVANCE => (m.trace.state, m.from),
        _ => (l.state, l.cell),
    }
}

/// What a request should reveal once its answer arrives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reveal {
    Start,
    /// Line by line: show the first line of the new move.
    Line,
    /// A whole move at once.
    Move,
    Episode,
}

/// What STEP LINE does next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineStep {
    /// Move the pointer to this line; the values are already known.
    Show(usize),
    /// Ask the simulation for the next move or episode.
    Request(LessonMode, Reveal),
}

pub fn step_line(lesson: Option<&LessonState>, line: usize) -> LineStep {
    let Some(l) = lesson else { return LineStep::Request(LessonMode::Start, Reveal::Start) };
    match &l.last {
        None => LineStep::Request(LessonMode::Move, Reveal::Line),
        Some(_) if line < ADVANCE => LineStep::Show((line + 1).max(CHOOSE)),
        Some(_) if line == ADVANCE && l.finished.is_some() => LineStep::Show(UNTIL),
        Some(_) => LineStep::Request(LessonMode::Move, Reveal::Line),
    }
}

/// The line to point at after a request's answer arrives.
pub fn line_after(reveal: Reveal, l: &LessonState) -> usize {
    match (reveal, &l.last) {
        (_, None) => RESET,
        (Reveal::Start | Reveal::Line, Some(_)) => CHOOSE,
        (Reveal::Move, Some(_)) if l.finished.is_none() => ADVANCE,
        (Reveal::Move | Reveal::Episode, Some(_)) => UNTIL,
    }
}

/// The world as it should look at `line`: before the move is taken, the raft
/// is still at s and no reward has flashed yet.
pub fn displayed_model(l: &LessonState, line: usize) -> RenderModel {
    let mut model = l.render.clone();
    match &l.last {
        Some(m) if line < ACT => {
            model.events.clear();
            if let Some(a) = model.agents.first_mut() {
                a.position = (m.from.0 as f32 + 0.5, m.from.1 as f32 + 0.5);
            }
        }
        None => model.events.clear(),
        Some(_) => {}
    }
    model
}

/// The Q-table as it should look at `line`: before the update line, the
/// updated value still shows its old number.
pub fn displayed_policy(policy: &[CellPolicy], l: &LessonState, line: usize) -> Vec<CellPolicy> {
    let mut out = policy.to_vec();
    if let Some(m) = &l.last
        && line < UPDATE
        && let Some(c) = out.iter_mut().find(|c| c.cell == m.from)
    {
        c.values[m.trace.action as usize] = m.trace.update.old_value;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{SimRequest, SimResponse, Snapshot};
    use crate::sim::SimCore;

    fn lesson(sim: &mut SimCore, mode: LessonMode) -> Snapshot {
        match sim.handle(SimRequest::Lesson { mode }) {
            SimResponse::Lesson(s) => *s,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn stepping_line_by_line_walks_the_pseudocode() {
        let mut sim = SimCore::default();
        sim.handle(SimRequest::Load { level_id: "l01_the_raft".into(), seed: 1 });
        assert_eq!(step_line(None, 0), LineStep::Request(LessonMode::Start, Reveal::Start));
        let s = lesson(&mut sim, LessonMode::Start);
        let l = s.lesson.unwrap();
        assert_eq!(line_after(Reveal::Start, &l), RESET);
        assert_eq!(step_line(Some(&l), RESET), LineStep::Request(LessonMode::Move, Reveal::Line));

        let s = lesson(&mut sim, LessonMode::Move);
        let l = s.lesson.unwrap();
        let config = s.training.as_ref().unwrap().config;
        let mut line = line_after(Reveal::Line, &l);
        assert_eq!(line, CHOOSE);
        let mut seen = vec![line];
        while let LineStep::Show(next) = step_line(Some(&l), line) {
            line = next;
            seen.push(line);
        }
        assert_eq!(seen, vec![CHOOSE, ACT, TARGET, UPDATE, ADVANCE]);

        // Values appear only once their line is reached, and match the update.
        let early = line_values(&l, CHOOSE, &config);
        assert!(!early[CHOOSE].is_empty() && early[ACT].is_empty() && early[UPDATE].is_empty());
        let all = line_values(&l, ADVANCE, &config);
        let t = &l.last.as_ref().unwrap().trace;
        assert!(all[UPDATE].ends_with(&num(t.update.new_value)), "{}", all[UPDATE]);
        assert_eq!(all[TARGET], format!("−1.00 + 0.95 · 0.00 = {}", num(t.update.target)));

        // Before the move the raft is drawn at s, with the old Q-value.
        let policy = &s.training.as_ref().unwrap().policy;
        let before = displayed_policy(policy, &l, TARGET);
        let after = displayed_policy(policy, &l, UPDATE);
        let at = |p: &[CellPolicy]| p.iter().find(|c| c.cell == (0, 0)).unwrap().values[t.action as usize];
        assert_eq!((at(&before), at(&after)), (t.update.old_value, t.update.new_value));
        assert_eq!(displayed_model(&l, CHOOSE).agents[0].position, (0.5, 0.5));
    }

    #[test]
    fn negative_operands_get_parentheses() {
        assert_eq!((operand(-0.75), operand(1.5), num(-0.001)), ("(−0.75)".into(), "1.50".into(), "0.00".into()));
    }

    #[test]
    fn a_finished_episode_shows_its_outcome_then_starts_the_next() {
        let mut sim = SimCore::default();
        sim.handle(SimRequest::Load { level_id: "l01_the_raft".into(), seed: 1 });
        let s = lesson(&mut sim, LessonMode::Episode);
        let l = s.lesson.unwrap();
        assert_eq!(line_after(Reveal::Episode, &l), UNTIL);
        let values = line_values(&l, UNTIL, &s.training.unwrap().config);
        assert!(values[UNTIL].contains("moves"), "{}", values[UNTIL]);
        assert_eq!(step_line(Some(&l), ADVANCE), LineStep::Show(UNTIL));
        assert_eq!(step_line(Some(&l), UNTIL), LineStep::Request(LessonMode::Move, Reveal::Line));
        let next = lesson(&mut sim, LessonMode::Move).lesson.unwrap();
        assert_eq!((next.episode, line_after(Reveal::Line, &next)), (2, RESET));
    }
}
