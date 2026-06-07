//! Compute worker of the bhtsne web demo.
//!
//! For now this hosts a trivial echo worker proving the two artifact build
//! (UI bundle plus dedicated worker bundle) and the gloo-worker message
//! channel. The t-SNE computation will replace it.

use gloo_worker::{HandlerId, Worker, WorkerScope};
use serde::{Deserialize, Serialize};

/// Message sent from the UI to the worker.
#[derive(Serialize, Deserialize, Debug)]
pub struct PingRequest {
    pub payload: String,
}

/// Message sent from the worker back to the UI.
#[derive(Serialize, Deserialize, Debug)]
pub struct PingResponse {
    pub payload: String,
}

/// Placeholder worker that echoes the payload back.
pub struct EchoWorker;

impl Worker for EchoWorker {
    type Message = ();
    type Input = PingRequest;
    type Output = PingResponse;

    fn create(_scope: &WorkerScope<Self>) -> Self {
        EchoWorker
    }

    fn update(&mut self, _scope: &WorkerScope<Self>, _msg: Self::Message) {}

    fn received(&mut self, scope: &WorkerScope<Self>, input: Self::Input, id: HandlerId) {
        scope.respond(
            id,
            PingResponse {
                payload: format!("worker echo: {}", input.payload),
            },
        );
    }
}
