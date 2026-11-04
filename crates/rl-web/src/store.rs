//! Trained rafts saved in the browser's IndexedDB, one per level, so a
//! reload or a return visit keeps the raft and its learning curve.
//!
//! Values are JSON strings of [`crate::protocol::SavedRaft`]. Failures are
//! not fatal: without storage the game still plays, it just forgets.

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{IdbDatabase, IdbOpenDbRequest, IdbRequest, IdbTransactionMode};

const DB_NAME: &str = "reinforcement-odyssey";
const DB_VERSION: u32 = 1;
const RAFTS: &str = "rafts";

/// Resolves when an IndexedDB request succeeds, with its result.
async fn wait(request: &IdbRequest) -> Result<JsValue, JsValue> {
    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        let done = request.clone();
        let on_success = Closure::once_into_js(move || {
            let _ = resolve.call1(&JsValue::UNDEFINED, &done.result().unwrap_or(JsValue::UNDEFINED));
        });
        let on_error = Closure::once_into_js(move || {
            let _ = reject.call0(&JsValue::UNDEFINED);
        });
        request.set_onsuccess(Some(on_success.unchecked_ref()));
        request.set_onerror(Some(on_error.unchecked_ref()));
    });
    JsFuture::from(promise).await
}

async fn open() -> Result<IdbDatabase, JsValue> {
    let factory = web_sys::window().ok_or("no window")?.indexed_db()?.ok_or("IndexedDB is not available")?;
    let request: IdbOpenDbRequest = factory.open_with_u32(DB_NAME, DB_VERSION)?;
    let on_upgrade = Closure::once_into_js(move |e: web_sys::Event| {
        let db = e
            .target()
            .and_then(|t| t.dyn_into::<IdbOpenDbRequest>().ok())
            .and_then(|r| r.result().ok())
            .map(JsCast::unchecked_into::<IdbDatabase>);
        if let Some(db) = db
            && !db.object_store_names().contains(RAFTS)
        {
            let _ = db.create_object_store(RAFTS);
        }
    });
    request.set_onupgradeneeded(Some(on_upgrade.unchecked_ref()));
    Ok(wait(&request).await?.unchecked_into())
}

async fn run(
    mode: IdbTransactionMode,
    op: impl FnOnce(&web_sys::IdbObjectStore) -> Result<IdbRequest, JsValue>,
) -> Result<JsValue, JsValue> {
    let db = open().await?;
    let store = db.transaction_with_str_and_mode(RAFTS, mode)?.object_store(RAFTS)?;
    let result = wait(&op(&store)?).await;
    db.close();
    result
}

/// The saved raft for a level, as JSON.
pub async fn load(level_id: &str) -> Option<String> {
    run(IdbTransactionMode::Readonly, |s| s.get(&JsValue::from_str(level_id))).await.ok()?.as_string()
}

pub async fn save(level_id: &str, json: &str) {
    let _ =
        run(IdbTransactionMode::Readwrite, |s| s.put_with_key(&JsValue::from_str(json), &JsValue::from_str(level_id)))
            .await;
}

pub async fn forget(level_id: &str) {
    let _ = run(IdbTransactionMode::Readwrite, |s| s.delete(&JsValue::from_str(level_id))).await;
}

pub async fn forget_all() {
    let _ = run(IdbTransactionMode::Readwrite, |s| s.clear()).await;
}
