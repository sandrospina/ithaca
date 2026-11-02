#[cfg(target_arch = "wasm32")]
fn main() {
    console_error_panic_hook::set_once();
    leptos::mount::mount_to_body(rl_web::app::App);
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("rl-web runs in the browser: `cd crates/rl-web && trunk serve`");
}
