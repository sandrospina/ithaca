# Testing the current build on other computers

Two ways to get the game onto another computer.

| | Address | When |
|---|---|---|
| **GitHub Pages** | https://sandrospina.github.io/ithaca/ | Every push to `main` deploys it, a few minutes later. Works anywhere. |
| **Local network** | `scripts/serve-lan.sh`, then the address it prints | A quick check before pushing. Same network only. |

The status bar shows which build is running, for example `MILESTONE 1 · BUILD b60d41b`. A `+` after the hash means
the build had uncommitted changes. Quote it in every report.

## GitHub Pages

- One-time setup: in the repository on GitHub, **Settings → Pages → Build and deployment → Source: GitHub Actions**.
- The workflow is `.github/workflows/pages.yml`. It runs the tests, builds the release client and publishes it. Its
  progress is under the repository's **Actions** tab. A failing test stops the deployment.
- To redeploy without a new commit: **Actions → Deploy to GitHub Pages → Run workflow**.

## Local network

```sh
scripts/serve-lan.sh              # build the release client, serve it on port 8081
scripts/serve-lan.sh --no-build   # serve the last build again
scripts/serve-lan.sh --port 9000
```

Needs Trunk and Python 3 on this Mac. The other computer must be on the same network. macOS asks once whether
Python may accept incoming connections: allow it. Ctrl-C stops it. Port 8080 stays free for `trunk serve`.

Plain `http` on a network address is not a secure context. The current build does not need one; WebGPU training
(Milestone 8) will, so test that on the Pages site.

## Saved progress is per address

Campaign progress (localStorage) and trained rafts (IndexedDB) belong to the address they were made on. Progress on
`localhost:8080`, on the network address and on the Pages site are three separate saves. **RESET PROGRESS** in the
status bar clears the one you are on.

## What to check

Browsers: Chrome, Edge, Firefox and Safari, on desktop. Phones and tablets are not supported yet. The layout needs a
window at least about 1200 × 800.

1. **Loads**: the voyage map, the arena and the briefing appear within a few seconds, with no error in the side panel.
2. **Level 0**: steer with the arrow keys or W A S D to the shore by the shortest route; the level completes.
3. **Level 1**: the first two tutorial steps show no phase buttons; ANALYSE appears at step 3 and CONFIGURE,
   TRAIN and DEPLOY at step 7.
   - ANALYSE: run a batch of 100 random voyages; the statistics and charts appear.
   - TRAIN: train to the budget; the learning curve rises and the arrows on the arena point to the shore.
   - STEP THROUGH: step lines with Space and moves with Enter; the code values change line by line.
   - DEPLOY: the trained raft takes the shortest route.
   - Reload the page: the trained raft comes back ("your saved raft is back").
4. **Level 2**: a reward edit in CONFIGURE changes what the trained raft does.
5. **Plain mode** (top right): the scanlines and glow go away and everything stays readable.

Shortcuts for testers:

- `?unlock=all` opens every level without completing the earlier ones.
- `?level=l02_the_lotus_eaters` starts on a given level.

Example: https://sandrospina.github.io/ithaca/?unlock=all&level=l01_the_raft

## Reporting a problem

Include the build (status bar), the browser and its version, the operating system, the level and view, what you
did, and what happened. A screenshot helps. For anything that looks like a crash, the browser's developer console
(F12) usually holds the error.
