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
];

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
}
