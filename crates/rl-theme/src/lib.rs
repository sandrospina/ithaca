//! Palette tokens and the glyph → colour mapping from `docs/visual-style.md`.
//!
//! The same tokens are mirrored as CSS custom properties in
//! `crates/rl-web/style.css`; a test keeps the two in sync.

use rl_core::Glyph;

pub struct Token {
    pub name: &'static str,
    pub hex: &'static str,
}

macro_rules! tokens {
    ($($ident:ident = $name:literal, $hex:literal;)*) => {
        $(pub const $ident: Token = Token { name: $name, hex: $hex };)*
        pub const ALL: &[Token] = &[$($ident),*];
    };
}

tokens! {
    SHELL_BG  = "shell-bg",  "#0B0906";
    SHELL_FG  = "shell-fg",  "#FFB000";
    SHELL_DIM = "shell-dim", "#7A5400";
    SHELL_HI  = "shell-hi",  "#FFE08A";
    ARENA_BG  = "arena-bg",  "#040806";
    TERRAIN   = "terrain",   "#33FF66";
    AGENT     = "agent",     "#4DE8FF";
    ADVERSARY = "adversary", "#FF3D9A";
    GOAL      = "goal",      "#FFD24D";
    HAZARD    = "hazard",    "#FF7A1A";
    POLICY    = "policy",    "#C8FFD8";
    // Comparison series, in fixed order. Validated for colour-blind
    // separation and contrast on the shell background; each series also has
    // its own dash pattern and letter.
    SERIES_A  = "series-a",  "#C88300";
    SERIES_B  = "series-b",  "#00A1A9";
    SERIES_C  = "series-c",  "#CB234A";
    SERIES_D  = "series-d",  "#8460D2";
}

/// The single colour each glyph is drawn in. Shape carries meaning first;
/// this colour is the second channel.
pub fn glyph_colour(glyph: Glyph) -> &'static str {
    match glyph {
        Glyph::Agent => AGENT.hex,
        Glyph::Adversary => ADVERSARY.hex,
        Glyph::Goal => GOAL.hex,
        Glyph::Terrain => TERRAIN.hex,
        Glyph::Hazard => HAZARD.hex,
        Glyph::Policy => POLICY.hex,
        // A lure is tempting by design: gold like the goal, told apart by its lotus shape.
        Glyph::Lure => GOAL.hex,
    }
}

/// Converts `#RRGGBB` plus an alpha in `[0, 1]` to a CSS `rgba(...)` string.
pub fn rgba(hex: &str, alpha: f64) -> String {
    let v = u32::from_str_radix(hex.trim_start_matches('#'), 16).unwrap_or(0);
    format!("rgba({}, {}, {}, {alpha:.3})", (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn css_mirrors_every_token() {
        let css = include_str!("../../rl-web/style.css");
        for t in ALL {
            let decl = format!("--{}: {};", t.name, t.hex);
            assert!(css.contains(&decl), "style.css is missing `{decl}`");
        }
    }

    #[test]
    fn rgba_converts() {
        assert_eq!(rgba("#FF8000", 0.5), "rgba(255, 128, 0, 0.500)");
    }
}
