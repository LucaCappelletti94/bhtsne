#!/usr/bin/env bash
# Builds the compute worker wasm bundle and places the wasm-bindgen output in
# the app assets, where the UI spawns it from (see WORKER_JS_URL in demo-app).
# Requires the wasm-bindgen CLI at the exact version pinned in the workspace
# Cargo.toml.
set -euo pipefail

cd "$(dirname "$0")/../.."

cargo build -p demo-worker --bin tsne_worker --target wasm32-unknown-unknown --release

wasm-bindgen \
    --target web \
    --no-typescript \
    --out-dir demo/app/assets/worker \
    --out-name tsne_worker \
    target/wasm32-unknown-unknown/release/tsne_worker.wasm

echo "worker bundle written to demo/app/assets/worker/"
