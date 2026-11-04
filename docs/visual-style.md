# Visual Style — Hybrid "Phosphor Terminal"

Decision (2026-11-02): **vector-phosphor arena** inside an **amber mainframe-terminal UI shell**.

The two halves have distinct roles:

| Layer | Era reference | Role |
|---|---|---|
| UI shell | 1970s–80s amber monochrome terminal (WOPR / VT220) | menus, Observation Builder, hyperparameters, logs, debugger, Calypso's dialogue |
| Arena | Vector arcade / oscilloscope (Asteroids, Battlezone, Tempest) | the world, agents, opponents, value overlays, replays |
| Charts | Oscilloscope with graticule | learning curves, loss, entropy, win rate |

## 1. The core rule: one RL concept, one visual form

Every RL concept has exactly one visual encoding, which never changes between levels.
A player who learns "cyan triangle = my agent" in Level 1 must be able to rely on it in Level 15.

| Concept | Shape | Colour token | Notes |
|---|---|---|---|
| Player agent (Odysseus / ship) | Triangle pointing along heading | `agent` cyan `#4DE8FF` | Teammates: same shape, dimmer outline |
| Adversary / opponent | Diamond or jagged polygon | `adversary` magenta `#FF3D9A` | Historical-policy adversaries get a numbered tag |
| Goal / objective | Circle with radiating ticks | `goal` gold `#FFD24D` | |
| Wall / terrain | Solid line segments | `terrain` green `#33FF66` | The default phosphor colour |
| Hazard / lethal cell | Cross-hatched fill | `hazard` orange `#FF7A1A` | Lethal = double hatch. Shape distinguishes it from adversary, not colour alone |
| Lure (e.g. the lotus field) | Lotus flower outline | `goal` gold `#FFD24D` | Tempting like the goal by design; told apart by shape |
| Reward event | Brief expanding ring | gold (positive) / magenta (negative) | Fades over ~300 ms |
| Q-value / state value | Cell brightness | diverging: green (+) / magenta (−) | Brightness ∝ magnitude, normalised per view |
| Policy | Thin arrow per cell / heading line | `policy` white-green `#C8FFD8` | Length or brightness ∝ action probability / Q margin |
| Agent observation (what it sees) | Crisp rendering | — | |
| Unobserved true state | Dim, noisy, scanline-covered | `fog` | Debug view only. Teaches state ≠ observation |
| Trajectory | Phosphor persistence trail | colour of the actor | Free by-product of frame fading |
| Messages (Level 14) | 16 glyphs (Greek letters Α–Π) | colour of the sender | Shown above the sender during replay |
| Resources (health, energy, ammo) | Segmented vector bars | per-resource token | Always in the same screen position |

Colour-blind safety: every distinction in the table above is carried by **shape or pattern** first and colour second.

## 2. Palette tokens

```text
shell.bg        #0B0906   near-black warm
shell.fg        #FFB000   amber
shell.dim       #7A5400   dim amber (disabled, secondary text)
shell.hi        #FFE08A   highlighted amber (selection, focus)

arena.bg        #040806   near-black green
terrain         #33FF66
agent           #4DE8FF
adversary       #FF3D9A
goal            #FFD24D
hazard          #FF7A1A
policy          #C8FFD8
fog             #33FF66 at 15% + noise
```

Tokens live in one Rust module (`theme`) consumed by every renderer and by the CSS (generated or mirrored as CSS custom properties). No renderer may hard-code a colour.

## 3. Typography

- Display / headings: **VT323** (OFL), terminal pixel style.
- Body, numbers, tables: **IBM Plex Mono** (OFL). Readability beats authenticity for Q-values and hyperparameters.
- Greek touches (level titles, Calypso's name) may use a Greek-key (meander) border drawn as vector lines, not a decorative font.

## 4. Effects

| Effect | Where | Implementation | Toggleable |
|---|---|---|---|
| Bloom / glow | Arena lines | Canvas2D `shadowBlur` first; WebGL2 post-process later | Yes |
| Phosphor persistence | Arena | Draw a translucent `arena.bg` rect each frame instead of clearing | Yes (also controls trail length) |
| Scanlines | Shell only, very subtle | CSS repeating gradient overlay | Yes |
| Screen curvature | Nowhere at first | — | — |
| Flicker | Never above 3 Hz | — | Off by default |

Accessibility settings: "Plain mode" disables all effects, "Reduced motion" disables trails and reward rings.

## 5. Renderer architecture rules

1. Simulation never knows about rendering. The renderer consumes a **render model** (positions, headings, cell overlays) built from environment state + agent internals.
2. Start with Canvas2D. Move to WebGL2 only when bloom or agent count requires it.
3. All coordinates are world units; one camera transform maps them to the canvas. The same renderer then serves grid levels and continuous levels.
4. Rendering is never on the training path.

## 6. Consistency safeguard

Build a **style-sheet scene** in Milestone 0: one static arena that renders every glyph, overlay and colour token side by side.
- It is the visual reference when adding new levels.
- Snapshot it in CI (headless browser screenshot) as a visual regression test.
