# Experimental HIP backend

Rust host code and HIP kernels for the supported AMD execution path. It shares
model formats and server infrastructure with `core`, but has its own placement,
graph execution, prefill and speculative decoding implementations.

```bash
cargo check --locked -p dsx-amd
# With a compatible ROCm installation:
cargo build --locked --release -p dsx-amd
./target/release/dsx-amd --help
```

Use `HIPCC` when the HIP compiler is outside the detected locations. Consult
`build.rs` for the current architecture target. Model files are supplied separately.
Development commands use local scratch files and must run in a suitable writable
environment. Historical CUDA measurements are not HIP performance claims.
