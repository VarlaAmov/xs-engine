# XS Engine

**Rust inference engine for DeepSeek · CUDA · experimental HIP backend**

XS Engine explores efficient inference for quantized mixture-of-experts models:
keep frequently used experts on the GPU, execute the rest across CPU and GPU,
and reuse conversation state instead of processing the same prompt again.

[Getting started](#getting-started) · [Architecture](docs/ARCHITECTURE.md) ·
[Experiments and results](docs/EXPERIMENTS.md) · [Contributing](CONTRIBUTING.md)

## What is here

- **Hybrid expert execution.** Adaptive GPU residency and CPU/GPU execution of cache misses.
- **Optimized CUDA paths.** Quantized dense projections, expert kernels and sparse indexing.
- **Conversation reuse.** Shared prompt snapshots, bounded in-memory caches and a disk archive.
- **Compatible APIs.** OpenAI-style chat and responses, Anthropic-style messages, streaming and tool calls.
- **Reasoning controls.** Optional thinking budgets, token penalties, min-p sampling and uncertainty statistics.
- **Reference implementations.** CPU numerics, encoding fixtures and kernel correctness checks.

This is an experimental engine specialized for the supported DeepSeek formats.
The CUDA and HIP backends have different capabilities; results from one backend
do not establish performance on the other. Model weights are supplied separately.

## Getting started

Use a Rust toolchain supporting edition 2024. CUDA builds require a compatible
CUDA toolkit; the current bindings target CUDA 13. Set `CUDA_ARCHS` to your GPU's
compute capability without a decimal point.

```bash
# CPU reference, server and encoding tests; no model download required.
cargo test --locked -p dsx-core -p dsx --no-default-features

# Example CUDA target: compute capability 8.6.
CUDA_ARCHS=86 cargo build --locked --release -p dsx --features cuda

./target/release/dsx-server \
  --model ./models/model.gguf \
  --engram ./models/engram.gguf \
  --tokenizer ./models/tokenizer.json \
  --host 127.0.0.1 --port 8090 --ctx 32768
```

Use matching model, Engram and tokenizer files. Consult `dsx-server --help` for
memory budgets, GPU placement, context limits, session caching and authentication.
The native CPU build configuration means binaries should be rebuilt for a different CPU.

```bash
curl http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"xs","messages":[{"role":"user","content":"Explain a hash table."}],"max_tokens":128}'
```

For HIP builds, see [the AMD backend](amd/README.md).

## Performance work that has already been done

The [experiment journal](docs/EXPERIMENTS.md) records successful changes,
rejected implementations and ideas that still need a valid experiment. Each entry
states the measured result, its limits and what must change before revisiting it.

Historical CUDA measurements include a decode improvement from **24.71 to 37.57
tokens/s** at a 32K input and a 190K prefill reduction from **404 to 213 seconds**.
These are results for a specific historical configuration, not a throughput guarantee.
Later experiments improved prompt reuse and recovery without a demonstrated decode gain.

Read the journal before repeating an optimization. In particular, a failed prefetch
implementation does not rule out every prefetch design, and an expensive verification
path does not establish that speculative decoding can never work.

## Repository map

| Directory | Purpose |
|---|---|
| `core/` | Model readers, CPU reference, formats, sampling, session management and HTTP APIs |
| `nvidia/` | CUDA backend, hybrid expert execution and GPU tests |
| `amd/` | Experimental HIP backend and model conversion utilities |
| `docs/` | Architecture and the experiment journal |

## License and attribution

The engine packages declare MIT licensing; see [LICENSE](LICENSE).
The embedded web UI is derived from llama.cpp and retains its
[upstream license](core/webui/LICENSE-llama.cpp). DeepSeek tokenizer fixtures retain
[their license](core/tests/data/LICENSE-tokenizer). Model weights have their own
licensing terms and are not included.
