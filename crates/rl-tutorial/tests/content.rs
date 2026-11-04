//! Every tutorial file must parse, validate, and belong to an existing level.

use std::path::PathBuf;

use rl_tutorial::TutorialDef;

#[test]
fn every_tutorial_file_is_valid_and_matches_a_level() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../content");
    let mut count = 0;
    for entry in std::fs::read_dir(root.join("tutorials")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "ron") {
            continue;
        }
        let source = std::fs::read_to_string(&path).unwrap();
        let def = TutorialDef::parse(&source).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(format!("{}.ron", def.id), path.file_name().unwrap().to_string_lossy());
        assert!(root.join("levels").join(format!("{}.ron", def.level)).exists(), "{}: unknown level", def.id);
        count += 1;
    }
    assert!(count >= 2);
}
