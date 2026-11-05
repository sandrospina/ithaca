# Burn hardware spike

One small Burn model, trained on every backend the game plans to use, with checks that say whether that backend
can be trusted. This is a hardware test, not deep-RL work. Results go in `docs/training-architecture.md`.

This directory is its own Cargo workspace, so Burn stays out of the game's build and test runs.

## What one run does

1. **Matmul check**: multiplies host-generated matrices on the device and compares with an f64 reference.
2. **Fit**: the game's own tabular Q-learning learns Level 1 (500 episodes), then a 25→64→64→4 network learns to
   reproduce that Q-table on the device (400 full-batch Adam steps). It passes when:
   - the final loss is below 0.05;
   - its best move agrees with the table on every state where the table has a clear favourite;
   - its greedy policy sails the shortest route (8 moves, return +2) on all 10 test seeds.
3. **Determinism**: the fit is run twice from the same seed and the predictions compared bit for bit.
4. **Benchmark**: 50 Adam steps of a 256→512→512→4 network at batch 4096 (batch 1024 and 10 steps on CPU backends),
   reported as steps/s and GFLOP/s. The final loss is recomputed on the host from the device's own predictions.
   A mismatch means the device computed something wrong.

`PASS` needs the matmul check, the fit and the loss recomputation to hold. Determinism is reported, not required.

## Running it

Rust 1.95 or newer. The first build takes about 5 minutes. The first GPU run compiles kernels, so compare the
warm fit time (the second run), not the first.

```sh
cd spikes/burn
cargo run --release --features <features> -- <devices> --json results.jsonl
```

| Machine | Command | Needs |
|---|---|---|
| DGX Spark (ARM64, CUDA) | `--features cuda,fusion -- cuda` | CUDA toolkit (NVRTC) on the path |
| | `--features vulkan,fusion -- vulkan` | Vulkan driver: the fallback |
| RTX 2080 PC (CUDA) | `--features cuda,fusion -- cuda` | CUDA toolkit |
| | `--features vulkan,fusion -- vulkan` | Vulkan driver: the fallback |
| RX 9070 XT, native Windows | `--features vulkan,fusion -- vulkan dx12 wgpu` | Rust with the MSVC toolchain; Adrenalin driver |
| RX 9070 XT, WSL2 | `--features rocm,fusion -- rocm` | ROCm 7.2.1+ in WSL2, Adrenalin 26.2.2+ |
| Any machine, CPU | `-- flex` | nothing: the default build |
| Browser | `trunk serve` in this directory, then open http://127.0.0.1:8090 | a browser with WebGPU |

On each server, run the GPU line twice more with `,autotune` added to the features and compare. Paste the
printed summary and the JSON lines into `docs/training-architecture.md`. Run `nvidia-smi` or `rocminfo` once too,
for the driver version: CUDA and ROCm runs do not name their adapter.

A device that fails to start is reported as `FAILED TO RUN` with the error, and the run continues with the next
device. The exit code is 1 if anything failed.

## Features

- Backends: `flex` (default, pure-Rust CPU), `cpu`, `wgpu`, `metal`, `vulkan`, `webgpu`, `cuda`, `rocm`.
  The `dx12` device needs the `wgpu` feature (any of `wgpu`, `metal`, `vulkan` enable it), on Windows.
- `fusion`: kernel fusion. Use it everywhere: no correctness cost seen, and the small fit runs 3× faster in the browser.
- `autotune`: benchmarks kernel variants and keeps the fastest. **Off in the browser** (see below). On native GPUs
  it is optional: about 20% more throughput on Apple Metal, but 10–20 s of tuning on first use.

## Known problem: autotune in the browser

With autotune on (Burn's default), WebGPU in Chrome gave wrong results in 4 of 5 runs: a negative mean-squared loss,
a device loss of 0.125 where the host recomputed 0.316, a network that disagreed with the table on 14 of 21 states,
and NaN. The cause, seen through the page's WebGPU error log: one
CubeCL matmul variant (`matmul_entry_…_lhs_size_2_rhs_size_2…`) fails WGSL validation in Chrome. Autotune times it,
it does no work, so it looks fastest and is sometimes chosen. With autotune off, every run was correct and
bit-deterministic. `index.html` therefore builds with `webgpu,fusion` and no autotune.

The page also logs WebGPU's own errors (`window.gpuEvents`). A healthy run shows one entry: the adapter.

On macOS, the `cuda` and `rocm` features do not build: the LLVM that Burn bundles expects Homebrew's zstd. Neither
backend runs on a Mac anyway.
