//! Stamps the build with its git commit, shown in the status bar so testers
//! can say which build they saw. `BUILD_COMMIT` in the environment overrides it.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    println!("cargo:rerun-if-env-changed=BUILD_COMMIT");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");
    let commit = std::env::var("BUILD_COMMIT").ok().or_else(|| {
        let hash = git(&["rev-parse", "--short=7", "HEAD"])?;
        // Uncommitted changes to tracked files: the build is not exactly this commit.
        let dirty = git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
        Some(if dirty { format!("{hash}+") } else { hash })
    });
    println!("cargo:rustc-env=BUILD_COMMIT={}", commit.unwrap_or_else(|| "dev".into()));
}
