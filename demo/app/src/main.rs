//! Dioxus UI of the bhtsne web demo.
//!
//! For now this is a skeleton that spawns the compute worker and exchanges a
//! ping message with it, proving the two artifact build and the message
//! channel end to end.

use std::rc::Rc;

use demo_worker::{EchoWorker, PingRequest, PingResponse};
use dioxus::prelude::*;
use gloo_worker::Spawnable;

/// Folder asset holding the wasm-bindgen output of the worker bundle, generated
/// by demo/scripts/build-worker.sh before the app build. A folder asset keeps
/// the inner file names, which gloo-worker relies on to derive the .wasm URL
/// from the .js one.
static WORKER_ASSETS: Asset = asset!("/assets/worker");

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    let response = use_signal(|| String::from("no response yet"));

    // The bridge owns the worker and must live across renders, so it is
    // created once. The callback writes the worker replies into the signal.
    // Signals are Copy, the rebinding makes the closure a plain Fn.
    let bridge = use_hook(|| {
        Rc::new(
            EchoWorker::spawner()
                .callback(move |PingResponse { payload }| {
                    let mut response = response;
                    response.set(payload);
                })
                .spawn(&format!("{WORKER_ASSETS}/tsne_worker.js")),
        )
    });

    rsx! {
        h1 { "bhtsne web demo skeleton" }
        button {
            id: "ping",
            onclick: move |_| {
                bridge.send(PingRequest {
                    payload: String::from("ping from the UI"),
                })
            },
            "Ping worker"
        }
        p { id: "response", "{response}" }
    }
}
