//! Level and tutorial files bundled into the app. The files themselves live
//! in `content/` and are validated by the content tests.

pub struct LevelEntry {
    pub id: &'static str,
    pub menu_label: &'static str,
    pub level_source: &'static str,
    pub tutorial_source: &'static str,
}

pub const LEVELS: &[LevelEntry] = &[
    LevelEntry {
        id: "l00_calypsos_loom",
        menu_label: "0 · Calypso's Loom",
        level_source: include_str!("../../../content/levels/l00_calypsos_loom.ron"),
        tutorial_source: include_str!("../../../content/tutorials/l00_calypsos_loom.ron"),
    },
    LevelEntry {
        id: "l01_the_raft",
        menu_label: "1 · The Raft",
        level_source: include_str!("../../../content/levels/l01_the_raft.ron"),
        tutorial_source: include_str!("../../../content/tutorials/l01_the_raft.ron"),
    },
    LevelEntry {
        id: "l02_the_lotus_eaters",
        menu_label: "2 · The Lotus-Eaters",
        level_source: include_str!("../../../content/levels/l02_the_lotus_eaters.ron"),
        tutorial_source: include_str!("../../../content/tutorials/l02_the_lotus_eaters.ron"),
    },
    LevelEntry {
        id: "l03_cyclops_cave",
        menu_label: "3 · The Cyclops' Cave",
        level_source: include_str!("../../../content/levels/l03_cyclops_cave.ron"),
        tutorial_source: include_str!("../../../content/tutorials/l03_cyclops_cave.ron"),
    },
];

/// The concept library (`content/concepts.ron`).
pub const CONCEPTS: &str = include_str!("../../../content/concepts.ron");

pub fn level(id: &str) -> Option<&'static LevelEntry> {
    LEVELS.iter().find(|l| l.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_content_is_valid_and_consistent() {
        for entry in LEVELS {
            let def = rl_envs::LevelDef::parse(entry.level_source).unwrap();
            assert_eq!(def.id, entry.id);
            rl_envs::load_world(entry.level_source).unwrap();
            let tutorial = rl_tutorial::TutorialDef::parse(entry.tutorial_source).unwrap();
            assert_eq!(tutorial.level, entry.id);
            assert_eq!(def.tutorial.as_deref(), Some(tutorial.id.as_str()));
        }
    }

    /// Tutorials run headlessly: every step points only at parts of the
    /// interface its level actually has. (Guided experiments run in
    /// `sim::tests::every_guided_experiment_shows_its_lesson`.)
    #[test]
    fn every_tutorial_points_at_what_its_level_has() {
        use rl_envs::Pilot;
        use rl_tutorial::UiElement as U;
        for entry in LEVELS {
            let def = rl_envs::LevelDef::parse(entry.level_source).unwrap();
            let tutorial = rl_tutorial::TutorialDef::parse(entry.tutorial_source).unwrap();
            let agent = def.pilot == Pilot::Agent;
            let has_batch = matches!(&def.challenge, Some(c) if matches!(&c.goal, rl_envs::GoalDef::Tasks(t)
                if t.iter().any(|t| matches!(t, rl_envs::TaskDef::BatchRun { .. }))));
            for (i, step) in tutorial.steps.iter().enumerate() {
                for el in step.highlight() {
                    let ok = match el {
                        U::Controls => !agent,
                        U::Autopilot
                        | U::Configure
                        | U::Train
                        | U::Deploy
                        | U::Walkthrough
                        | U::QValues
                        | U::Compare
                        | U::Analyse => agent,
                        U::BestSeed | U::Quiz => agent && has_batch,
                        U::Rewards => def.player_editable.reward.is_some(),
                        U::Observation => def.player_editable.observation.is_some(),
                        U::Objective => def.challenge.is_some(),
                        _ => true,
                    };
                    assert!(ok, "{} step {i} highlights {el:?}, which this level does not have", entry.id);
                }
                if step.experiment().is_some() {
                    assert!(agent, "{} step {i}: experiments need an agent", entry.id);
                }
            }
        }
    }

    #[test]
    fn concepts_point_at_the_steps_that_teach_them() {
        let lib = rl_tutorial::ConceptLibrary::parse(CONCEPTS).unwrap();
        assert!(lib.concepts.len() >= 20);
        let mut last_level = 0;
        for c in &lib.concepts {
            let index =
                LEVELS.iter().position(|l| l.id == c.level).unwrap_or_else(|| panic!("{}: unknown level", c.id));
            assert!(index >= last_level, "{}: concepts are listed in level order", c.id);
            last_level = index;
            let tutorial = rl_tutorial::TutorialDef::parse(LEVELS[index].tutorial_source).unwrap();
            assert!(c.step < tutorial.steps.len(), "{}: step {} is past the tutorial's end", c.id, c.step);
        }
    }

    /// `docs/rl-concepts.md` is generated from the concept library. Run with
    /// `UPDATE_DOCS=1` to rewrite it after changing `content/concepts.ron`.
    #[test]
    fn concept_docs_are_up_to_date() {
        let lib = rl_tutorial::ConceptLibrary::parse(CONCEPTS).unwrap();
        let markdown = lib.to_markdown(|id| format!("Level {}", level(id).expect("checked").menu_label));
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/rl-concepts.md");
        if std::env::var_os("UPDATE_DOCS").is_some() {
            std::fs::write(&path, &markdown).unwrap();
        }
        let current = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(
            current == markdown,
            "docs/rl-concepts.md is stale: run `UPDATE_DOCS=1 cargo test -p rl-web concept_docs`"
        );
    }
}
