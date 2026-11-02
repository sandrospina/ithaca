//! Web Worker entry point: the simulation runs here, off the UI thread.

#[cfg(target_arch = "wasm32")]
fn main() {
    use gloo_worker::Registrable;
    console_error_panic_hook::set_once();
    rl_web::worker::SimWorker::registrar().register();
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {}
