//! Entry point of the dedicated worker wasm bundle.

use demo_worker::EchoWorker;
use gloo_worker::Registrable;

fn main() {
    console_error_panic_hook::set_once();

    EchoWorker::registrar().register();
}
