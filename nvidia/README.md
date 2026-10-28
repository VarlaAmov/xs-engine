# CUDA backend

Rust host code, custom CUDA kernels and hybrid CPU/GPU expert execution.
Dense projections, expert residency, peer transfers and sparse indexing are
implemented here; shared formats and APIs come from `core`.

```bash
CUDA_ARCHS=86 cargo build --locked --release -p dsx --features cuda
cargo test --locked -p dsx-core -p dsx --no-default-features
```

Choose `CUDA_ARCHS` for the target GPU. See `dsx --help` and `dsx-server --help`
for reference execution, evaluation, generation and serving commands.
GPU tests need the matching toolkit and hardware; ignored full-model tests also
need an exclusive test environment and model paths.

See [architecture](../docs/ARCHITECTURE.md) and
[experiments](../docs/EXPERIMENTS.md) before changing a performance path.
