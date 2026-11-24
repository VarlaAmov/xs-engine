# Architecture

**Reviewed:** 2026-10-28 · **Status:** current

## Model and execution

`core` reads GGUF and safetensors, implements block decoders and the CPU reference,
and shares the server infrastructure between the CUDA and HIP backends.

The CUDA path places dense weights and a subset of experts in GPU memory.
An adaptive residency policy replaces experts as routing patterns change.
For a missing expert, the hybrid path splits intermediate rows between CPU and
GPU execution; GPU slices can read pinned host memory directly. Peer-capable GPUs
can exchange inputs without staging through the CPU.

Quantized dense kernels reduce weight decoding overhead. The sparse indexer
selects candidate positions before attention. Prefill groups work over tokens;
decode follows a single sequence through the layers. Changes to batching or expert
placement can change floating-point accumulation order, so correctness must be
evaluated independently of speed.

## Conversation state

A live conversation can be saved in a bounded host-memory cache and then in a disk
archive. Shared prompt snapshots avoid reprocessing a common system prefix.
Disk state has integrity checks and compatibility metadata; writes use a temporary
file followed by synchronization and rename. An unreadable snapshot falls back to
processing the prompt again.

Cancellation requires a reusable rollback point before the final assistant header.
Tokenization may change the tail of a previous user message when adjacent user
messages are combined. The rollback margin accounts for the cases covered by the
acceptance tests; arbitrary earlier edits can still require a longer prefill.

## APIs and controls

The shared server supports streaming, tools, sampling and optional token statistics.
Reasoning budgets, hesitation penalties, min-p and short-answer uncertainty flags
are explicit controls. Uncertainty thresholds are calibrated heuristics for a
limited evaluation set, not a general fact checker.

Experimental memory packing, dynamic context allocation and prefetch paths are
controlled separately. Their existence in the source does not mean they are enabled
or beneficial for a particular deployment. See the experiment journal for decisions.

## Backends

The CUDA backend uses `cudarc`, custom CUDA kernels and hybrid CPU/GPU experts.
The experimental HIP backend has its own kernels, graph execution and draft path.
Memory placement, accepted draft lengths and throughput do not transfer directly
between the backends.
