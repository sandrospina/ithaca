//! Burn hardware spike: the same small model, trained on whichever backend
//! this build enables, with checks that say whether the backend is usable.
//!
//! The task is tied to the game. The game's own tabular Q-learning learns
//! Level 1, then a small network learns to reproduce that Q-table (one-hot
//! cell in, four Q-values out). The network passes when its greedy policy
//! sails the same shortest route. A second, larger network measures
//! training throughput.
//!
//! Everything is async so that the browser build, where GPU reads are
//! asynchronous, runs the same code as the native one.

use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::Distribution;
use rl_agents::{GreedyPolicy, QLearner, QLearningConfig, QTable};
use rl_core::{Rng, World};
use rl_training::{EpisodeOutcome, play_episode, train_episode};
use serde::Serialize;
use web_time::Instant;

const RAFT: &str = include_str!("../../../content/levels/l01_the_raft.ron");
const ACTIONS: usize = 4;
/// Episodes of tabular Q-learning behind the target table.
const TABULAR_EPISODES: u32 = 500;
/// Full-batch Adam steps for the fit.
const FIT_STEPS: usize = 400;
const FIT_HIDDEN: usize = 64;
/// The shortest route on Level 1: eight moves, return +2.
const SHORTEST: (u32, f32) = (8, 2.0);

#[derive(Module, Debug)]
pub struct Mlp {
    l1: Linear,
    l2: Linear,
    l3: Linear,
    act: Relu,
}

impl Mlp {
    pub fn new(input: usize, hidden: usize, output: usize, device: &Device) -> Self {
        Self {
            l1: LinearConfig::new(input, hidden).init(device),
            l2: LinearConfig::new(hidden, hidden).init(device),
            l3: LinearConfig::new(hidden, output).init(device),
            act: Relu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let x = self.act.forward(self.l1.forward(x));
        let x = self.act.forward(self.l2.forward(x));
        self.l3.forward(x)
    }
}

fn mse(pred: Tensor<2>, target: Tensor<2>) -> Tensor<1> {
    (pred - target).powf_scalar(2.0).mean()
}

async fn read(t: Tensor<2>) -> Result<Vec<f32>, String> {
    let data = t.into_data_async().await.map_err(|e| format!("{e:?}"))?;
    data.try_to_vec::<f32>().map_err(|e| format!("{e:?}"))
}

async fn read_scalar(t: Tensor<1>) -> Result<f32, String> {
    let data = t.into_data_async().await.map_err(|e| format!("{e:?}"))?;
    data.try_to_vec::<f32>().map_err(|e| format!("{e:?}"))?.first().copied().ok_or_else(|| "empty tensor".into())
}

/// The Q-table the game's tabular Q-learning learns on Level 1, with the default settings.
pub fn tabular_target() -> (rl_envs::GridWorld, QTable) {
    let mut world = rl_envs::load_world(RAFT).expect("Level 1 loads");
    let states = world.spec().agents[0].observation.state_count().expect("discrete states") as u32;
    let mut learner = QLearner::new(states, ACTIONS as u32, QLearningConfig::default());
    let mut rng = Rng::new(7);
    for e in 0..TABULAR_EPISODES {
        train_episode(&mut world, &mut learner, u64::from(e), &mut rng);
    }
    (world, learner.table)
}

#[derive(Clone, Debug, Serialize)]
pub struct FitReport {
    pub states: usize,
    pub steps: usize,
    pub seconds: f64,
    pub final_loss: f32,
    /// States whose table row has one clear best move, and how many of those the network agrees on.
    pub clear_states: usize,
    pub agreeing: usize,
    /// Test voyages the network's greedy policy sails on the shortest route.
    pub shortest_routes: u32,
    pub tested: u32,
    /// The network's predictions, for the determinism check.
    #[serde(skip)]
    pub predictions: Vec<f32>,
}

/// Trains the small network to reproduce the tabular Q-values.
pub async fn fit(device: &Device, seed: u64) -> Result<FitReport, String> {
    let (mut world, table) = tabular_target();
    let states = table.states as usize;
    let mut inputs = vec![0.0f32; states * states];
    let mut targets = Vec::with_capacity(states * ACTIONS);
    for s in 0..states {
        inputs[s * states + s] = 1.0;
        targets.extend_from_slice(table.row(s as u32));
    }

    let device = device.clone().autodiff();
    device.seed(seed);
    let x = Tensor::<2>::from_data(TensorData::new(inputs, [states, states]), &device);
    let y = Tensor::<2>::from_data(TensorData::new(targets, [states, ACTIONS]), &device);
    let mut model = Mlp::new(states, FIT_HIDDEN, ACTIONS, &device);
    let mut optim = AdamConfig::new().init();

    let start = Instant::now();
    for _ in 0..FIT_STEPS {
        let loss = mse(model.forward(x.clone()), y.clone());
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(1e-2, model, grads);
    }
    let final_loss = read_scalar(mse(model.forward(x.clone()), y)).await?;
    let predictions = read(model.forward(x)).await?;
    let seconds = start.elapsed().as_secs_f64();

    // Best-move agreement, on states where the table has a clear favourite.
    let (mut clear_states, mut agreeing) = (0, 0);
    let mut distilled = QTable::new(states as u32, ACTIONS as u32);
    for s in 0..states {
        let row = table.row(s as u32);
        let net = &predictions[s * ACTIONS..(s + 1) * ACTIONS];
        for (a, &v) in net.iter().enumerate() {
            distilled.set(s as u32, a as u32, v);
        }
        let mut sorted = row.to_vec();
        sorted.sort_by(|a, b| b.total_cmp(a));
        if sorted[0] - sorted[1] > 0.05 {
            clear_states += 1;
            if argmax(row) == argmax(net) {
                agreeing += 1;
            }
        }
    }

    // The network's greedy policy on the level's test seeds.
    let mut rng = Rng::new(1);
    let seeds = 1000..1010u64;
    let tested = seeds.clone().count() as u32;
    let shortest_routes = seeds
        .filter(|&seed| {
            let ep = play_episode(&mut world, &mut GreedyPolicy(&distilled), seed, &mut rng);
            ep.outcome == EpisodeOutcome::ReachedGoal && (ep.length, ep.episode_return) == SHORTEST
        })
        .count() as u32;

    Ok(FitReport {
        states,
        steps: FIT_STEPS,
        seconds,
        final_loss,
        clear_states,
        agreeing,
        shortest_routes,
        tested,
        predictions,
    })
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i)
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct MatmulCheck {
    pub size: [usize; 3],
    /// Largest error relative to the largest reference value.
    pub max_relative_error: f64,
    pub ok: bool,
}

/// Multiplies host-generated matrices on the device and compares with a
/// reference computed here in f64.
pub async fn matmul_check(device: &Device) -> Result<MatmulCheck, String> {
    let (m, k, n) = (192, 320, 160);
    let mut rng = Rng::new(5);
    let mut fill = |len: usize| (0..len).map(|_| rng.next_f32() * 2.0 - 1.0).collect::<Vec<f32>>();
    let a = fill(m * k);
    let b = fill(k * n);
    let ta = Tensor::<2>::from_data(TensorData::new(a.clone(), [m, k]), device);
    let tb = Tensor::<2>::from_data(TensorData::new(b.clone(), [k, n]), device);
    let got = read(ta.matmul(tb)).await?;
    let mut reference = vec![0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = f64::from(a[i * k + p]);
            for j in 0..n {
                reference[i * n + j] += av * f64::from(b[p * n + j]);
            }
        }
    }
    let scale = reference.iter().fold(0f64, |acc, v| acc.max(v.abs()));
    let max_err = got.iter().zip(&reference).fold(0f64, |acc, (g, r)| acc.max((f64::from(*g) - r).abs()));
    let max_relative_error = max_err / scale;
    Ok(MatmulCheck { size: [m, k, n], max_relative_error, ok: got.len() == m * n && max_relative_error < 1e-4 })
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct BenchConfig {
    pub input: usize,
    pub hidden: usize,
    pub batch: usize,
    pub steps: usize,
}

impl BenchConfig {
    /// About the size of a DQN for the later levels, at a generous batch.
    pub const STANDARD: BenchConfig = BenchConfig { input: 256, hidden: 512, batch: 4096, steps: 50 };
    /// For slow CPU backends.
    pub const SMALL: BenchConfig = BenchConfig { input: 256, hidden: 512, batch: 1024, steps: 10 };

    /// Floating-point operations per training step: forward 2·b·Σ(in·out), backward twice that.
    fn flops_per_step(&self) -> f64 {
        let (i, h, b) = (self.input as f64, self.hidden as f64, self.batch as f64);
        6.0 * b * (i * h + h * h + h * ACTIONS as f64)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct BenchReport {
    pub config: BenchConfig,
    pub warmup_seconds: f64,
    pub seconds: f64,
    pub steps_per_second: f64,
    pub gflops: f64,
    pub final_loss: f32,
    /// The loss after training, from the device and recomputed on the host.
    pub device_loss: f32,
    pub host_loss: f32,
    pub loss_matches: bool,
}

/// Times full training steps of a larger network on random data. The first
/// steps compile kernels and are timed separately.
pub async fn bench(device: &Device, seed: u64, config: BenchConfig) -> Result<BenchReport, String> {
    let device = device.clone().autodiff();
    device.seed(seed);
    let x = Tensor::<2>::random([config.batch, config.input], Distribution::Default, &device);
    let y = Tensor::<2>::random([config.batch, ACTIONS], Distribution::Default, &device);
    let mut model = Mlp::new(config.input, config.hidden, ACTIONS, &device);
    let mut optim = AdamConfig::new().init();
    let mut step = |model: Mlp| {
        let loss = mse(model.forward(x.clone()), y.clone());
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let model = optim.step(1e-3, model, grads);
        (model, loss)
    };

    let start = Instant::now();
    for _ in 0..3 {
        model = step(model).0;
    }
    let (m, loss) = step(model);
    model = m;
    read_scalar(loss.detach()).await?;
    let warmup_seconds = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let mut last = None;
    for _ in 0..config.steps {
        let (m, loss) = step(model);
        model = m;
        last = Some(loss);
    }
    let final_loss = match last {
        Some(loss) => read_scalar(loss.detach()).await?,
        None => f32::NAN,
    };
    let seconds = start.elapsed().as_secs_f64();
    let steps_per_second = config.steps as f64 / seconds;

    // The device's loss against the same loss computed here from its own predictions.
    let device_loss = read_scalar(mse(model.forward(x.clone()), y.clone()).detach()).await?;
    let pred = read(model.forward(x).detach()).await?;
    let target = read(y).await?;
    let host_loss =
        (pred.iter().zip(&target).map(|(p, t)| f64::from(p - t).powi(2)).sum::<f64>() / pred.len() as f64) as f32;
    let loss_matches = (device_loss - host_loss).abs() <= 1e-4 * host_loss.abs().max(1e-3);
    Ok(BenchReport {
        config,
        warmup_seconds,
        seconds,
        steps_per_second,
        gflops: config.flops_per_step() * steps_per_second / 1e9,
        final_loss,
        device_loss,
        host_loss,
        loss_matches,
    })
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub backend: String,
    pub adapter: String,
    pub burn_version: &'static str,
    pub matmul: MatmulCheck,
    pub fit: FitReport,
    /// Whether a second fit from the same seed gave bit-identical predictions.
    pub deterministic: bool,
    /// The second fit's time: the first one includes compiling the GPU kernels.
    pub fit_warm_seconds: f64,
    pub bench: Option<BenchReport>,
    pub passed: bool,
}

/// The whole spike on one device: fit twice (for determinism), then benchmark.
pub async fn run(
    device: &Device,
    backend: &str,
    adapter: &str,
    bench_config: Option<BenchConfig>,
) -> Result<Report, String> {
    let matmul = matmul_check(device).await?;
    let fit_a = fit(device, 42).await?;
    let fit_b = fit(device, 42).await?;
    let deterministic = fit_a.predictions.len() == fit_b.predictions.len()
        && fit_a.predictions.iter().zip(&fit_b.predictions).all(|(a, b)| a.to_bits() == b.to_bits());
    let bench = match bench_config {
        Some(c) => Some(bench(device, 42, c).await?),
        None => None,
    };
    let passed = matmul.ok
        && bench.as_ref().is_none_or(|b| b.loss_matches)
        && fit_a.final_loss.is_finite()
        && fit_a.final_loss < 0.05
        && fit_a.agreeing == fit_a.clear_states
        && fit_a.shortest_routes == fit_a.tested;
    Ok(Report {
        backend: backend.to_string(),
        adapter: adapter.to_string(),
        burn_version: BURN_VERSION,
        matmul,
        fit: fit_a,
        deterministic,
        fit_warm_seconds: fit_b.seconds,
        bench,
        passed,
    })
}

pub const BURN_VERSION: &str = "0.22.1";

impl Report {
    pub fn summary(&self) -> String {
        let f = &self.fit;
        let mut s = format!(
            "{} · {}\n  matmul {}×{}×{}: max relative error {:.1e} · {}\n  fit: loss {:.5} after {} steps in {:.2}s ({:.2}s warm) · best move agrees on {}/{} clear states · \
             shortest route on {}/{} test voyages\n  deterministic: {}",
            self.backend,
            self.adapter,
            self.matmul.size[0],
            self.matmul.size[1],
            self.matmul.size[2],
            self.matmul.max_relative_error,
            if self.matmul.ok { "ok" } else { "WRONG" },
            f.final_loss,
            f.steps,
            f.seconds,
            self.fit_warm_seconds,
            f.agreeing,
            f.clear_states,
            f.shortest_routes,
            f.tested,
            if self.deterministic { "yes (bit-identical)" } else { "no" },
        );
        if let Some(b) = &self.bench {
            s += &format!(
                "\n  bench: {}→{}→{}→4, batch {}: {:.1} steps/s · {:.1} GFLOP/s · warm-up {:.2}s\n  \
                 loss on device {:.6} · recomputed on host {:.6} · {}",
                b.config.input,
                b.config.hidden,
                b.config.hidden,
                b.config.batch,
                b.steps_per_second,
                b.gflops,
                b.warmup_seconds,
                b.device_loss,
                b.host_loss,
                if b.loss_matches { "match" } else { "MISMATCH" },
            );
        }
        s += &format!("\n  {}", if self.passed { "PASS" } else { "FAIL" });
        s
    }
}

/// The browser build: runs the spike on the page's WebGPU adapter and prints the report.
#[cfg(target_arch = "wasm32")]
pub mod web {
    use burn::prelude::*;
    use burn::tensor::wgpu::WgpuBackend;

    fn show(text: &str) {
        web_sys::console::log_1(&text.into());
        let doc = web_sys::window().and_then(|w| w.document());
        if let Some(out) = doc.and_then(|d| d.get_element_by_id("out")) {
            out.set_text_content(Some(&format!("{}{text}\n", out.text_content().unwrap_or_default())));
        }
    }

    pub async fn run_in_page() {
        show(&format!("burn-spike · Burn {} · browser WebGPU", super::BURN_VERSION));
        let init = Device::wgpu_options().graphics_api(WgpuBackend::WebGpu).init_with_setup_async().await;
        let (device, setup) = match init {
            Ok(d) => d,
            Err(e) => {
                show(&format!("webgpu · FAILED TO RUN\n  {e:?}"));
                return;
            }
        };
        let info = setup.adapter.get_info();
        let adapter = format!("{} {} · {:?}", info.vendor, info.name, info.backend);
        // Dropping these handles destroys the GPU device, and wgpu's device-lost
        // handler then panics. The page lives as long as the device should.
        std::mem::forget(setup);
        show("running…");
        match super::run(&device, "webgpu", &adapter, Some(super::BenchConfig::STANDARD)).await {
            Ok(r) => {
                show(&r.summary());
                show(&format!("JSON {}", serde_json::to_string(&r).unwrap_or_default()));
            }
            Err(e) => show(&format!("webgpu · FAILED TO RUN\n  {e}")),
        }
        show("DONE");
    }
}
