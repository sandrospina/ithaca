//! Runs the spike on one or more devices:
//!
//! ```text
//! cargo run --release --features cuda -- cuda
//! cargo run --release --features wgpu -- vulkan dx12 --json results.jsonl
//! ```
//!
//! Devices: flex, cpu, wgpu, metal, vulkan, dx12, cuda, rocm, each needing its
//! Cargo feature (metal, vulkan and dx12 need `wgpu`). With no device named,
//! every device this build enables is tried.

#[cfg(target_arch = "wasm32")]
fn main() {
    wasm_bindgen_futures::spawn_local(burn_spike::web::run_in_page());
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    native::main();
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::io::Write;

    use burn::prelude::*;
    use burn_spike::{BenchConfig, Report, run};

    /// Devices this build can construct, in the order they are tried.
    fn enabled() -> Vec<&'static str> {
        let mut v = vec![];
        if cfg!(feature = "cuda") {
            v.push("cuda");
        }
        if cfg!(feature = "rocm") {
            v.push("rocm");
        }
        if cfg!(feature = "wgpu") {
            v.push("wgpu");
        }
        if cfg!(feature = "cpu") {
            v.push("cpu");
        }
        if cfg!(feature = "flex") {
            v.push("flex");
        }
        v
    }

    /// Builds the named device and describes its adapter.
    fn device(name: &str) -> Result<(Device, String), String> {
        match name {
            #[cfg(feature = "flex")]
            "flex" => Ok((Device::flex(), "pure-Rust CPU".into())),
            #[cfg(feature = "cpu")]
            "cpu" => Ok((Device::cpu(), "CubeCL CPU".into())),
            #[cfg(feature = "cuda")]
            "cuda" => Ok((Device::cuda(0), "CUDA device 0".into())),
            #[cfg(feature = "rocm")]
            "rocm" => Ok((Device::rocm(0), "ROCm/HIP device 0".into())),
            #[cfg(feature = "wgpu")]
            "wgpu" | "metal" | "vulkan" | "dx12" => {
                use burn::tensor::wgpu::WgpuBackend;
                let api = match name {
                    "metal" => WgpuBackend::Metal,
                    "vulkan" => WgpuBackend::Vulkan,
                    "dx12" => WgpuBackend::Dx12,
                    _ => WgpuBackend::Auto,
                };
                let (device, setup) =
                    Device::wgpu_options().graphics_api(api).init_with_setup().map_err(|e| format!("{e:?}"))?;
                let info = setup.adapter.get_info();
                Ok((
                    device,
                    format!("{} · {:?} · driver {} {}", info.name, info.backend, info.driver, info.driver_info),
                ))
            }
            other => Err(format!("device `{other}` is not enabled in this build (enabled: {})", enabled().join(", "))),
        }
    }

    pub fn main() {
        let mut names = vec![];
        let mut json = None;
        let mut bench = Some(BenchConfig::STANDARD);
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--json" => json = args.next(),
                "--no-bench" => bench = None,
                "--small-bench" => bench = Some(BenchConfig::SMALL),
                "-h" | "--help" => {
                    println!("usage: burn-spike [DEVICE...] [--json FILE] [--no-bench | --small-bench]");
                    println!("devices in this build: {}", enabled().join(", "));
                    return;
                }
                d => names.push(d.to_string()),
            }
        }
        if names.is_empty() {
            names = enabled().into_iter().map(String::from).collect();
        }

        println!(
            "burn-spike · Burn {} · {} {}",
            burn_spike::BURN_VERSION,
            std::env::consts::OS,
            std::env::consts::ARCH
        );
        let mut failed = false;
        for name in &names {
            // CPU backends get the small benchmark: the standard one takes minutes there.
            let config = bench.map(|b| if matches!(name.as_str(), "flex" | "cpu") { BenchConfig::SMALL } else { b });
            let result = std::panic::catch_unwind(|| {
                let (device, adapter) = device(name)?;
                pollster::block_on(run(&device, name, &adapter, config))
            });
            let report: Result<Report, String> = match result {
                Ok(r) => r,
                Err(p) => Err(p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "panicked".into())),
            };
            match report {
                Ok(r) => {
                    println!("\n{}", r.summary());
                    failed |= !r.passed;
                    if let Some(path) = &json {
                        let mut f =
                            std::fs::OpenOptions::new().create(true).append(true).open(path).expect("open --json file");
                        writeln!(f, "{}", serde_json::to_string(&r).expect("serialise report")).expect("write report");
                    }
                }
                Err(e) => {
                    println!("\n{name} · FAILED TO RUN\n  {e}");
                    failed = true;
                }
            }
        }
        std::process::exit(i32::from(failed));
    }
}
