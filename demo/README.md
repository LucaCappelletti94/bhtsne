# bhtsne web demo

A web demo that embeds dropped CSV, TSV or Parquet data with PCA followed by Barnes-Hut t-SNE, entirely in the browser. The computation runs in a dedicated web worker and the scatter plot animates as epochs progress.

## Architecture

Two wasm artifacts, kept as separate workspace members:

- `app`: the Dioxus web UI, built and served with `dx`.
- `worker`: the compute crate, compiled to its own wasm bundle and run as a dedicated web worker through `gloo-worker`. Depends on `bhtsne` with the `wasm_js` feature for real entropy in the browser.

## Development

Requirements: the `wasm32-unknown-unknown` target, the `dx` CLI (Dioxus 0.7) and the `wasm-bindgen` CLI at the exact version pinned in the workspace `Cargo.toml`.

From the repository root:

```
./demo/scripts/build-worker.sh   # build the worker bundle into demo/app/assets/worker/
dx serve -p demo-app             # build and serve the UI
```

The worker bundle is not rebuilt by `dx serve`, rerun the script after changing the `worker` crate.
