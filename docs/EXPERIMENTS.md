# Experiments and results

**Reviewed:** 2026-10-28 · **Status:** current

This journal summarizes historical engineering measurements. Deployment identifiers,
private paths, service configuration and conversation logs are intentionally absent.
The original raw runs are not included in this public snapshot; numerical results
below are historical summaries, not independently reproducible benchmark claims.

The main series used the CUDA backend with DeepSeek V4.1 Flash 1017 GSQ-RCO IQ2_XS,
hybrid CPU/GPU experts and a constrained GPU-resident expert cache. Results compare
variants within that series. They are not guarantees on other systems, model formats
or contexts. Input sizes are in tokens; K means approximately one thousand unless
an exact count is given.

## Accepted changes

| ID | Experiment | Recorded result | Decision and limitation |
|---|---|---|---|
| A01 | Quantized dense GEMV, hybrid expert misses and sparse indexer | At 32K, baseline 24.71 → 37.57 tokens/s after the combined changes. At 190K, 18.83 → 31.66 tokens/s; prefill 404 → 213 s. | Retained. These combined results do not isolate every individual optimization. |
| A02 | Shared prompt snapshots | Reusing a roughly 16K system prefix for a new conversation reduced first-token latency from 9.7 to 0.49 s; identical-prompt retry was 0.14 s. | Retained. Later rollback changes add a small replay tail; these exact timings belong to the earlier implementation. |
| A03 | Disk conversation archive | A roughly 50K conversation resumed after a process restart in 0.96 s instead of a roughly 50 s full prefill. A controlled snapshot restored the same continuation tokens. | Retained. Model startup time is separate from request latency; archive writes can delay another request. |
| A04 | Prefill chunk size 8192 → 4096 | 36.25 → 37.37 tokens/s (+3.1%); prefill 764 → 737 tokens/s. | Retained in the measured configuration. Trades a small prefill regression for more expert residency. |
| A05 | Earlier adaptive-cache planning | 36.13 → 37.18 tokens/s (+2.9%); an eight-sequence evaluation matched with the flag off/on. | Retained in the measured configuration. Combined with A04, API measurements improved by about 6–7%. |
| A06 | Protect a selected disk session during archive cleanup | Regression test reproduced deletion of the selected session; the protected session now survives. Engine-thread failure is reflected in health and process exit. | Retained as a reliability fix, without a decode-speed claim. |
| A07 | Cancellation rollback before the assistant header, with a token margin | A next user turn without the cancelled assistant: 9.26 → 0.64 s at roughly 11K. After cancellation, one case reprocessed 22 of 13,938 tokens. | Retained. Exact retry became 0.13 → 0.27 s in one API comparison because more tail tokens are replayed. |
| A08 | Adaptive CPU-window cost for short prefill (`XS_PREFILL_ADAPT`) | Adding 512 tokens: 1.75 → 1.36 s. One fresh-input 32K run: 36.4 → 31.6 s. | Retained in the measured configuration. Decode gain was not established; small differences need repeated, identical-input comparisons. |

## Implementations that did not pay off

Do not repeat these implementations unchanged. Revisit only when the stated design,
workload or bottleneck has changed.

| ID | Experiment | Recorded result | Decision / reason to revisit |
|---|---|---|---|
| R01 | Change expert-miss split from CPU/main/aux `1,4,4` | Baseline 37.86–38.29 tokens/s; `2,4,3` gave 37.57; `1,5,3` / `1,6,2` gave about 30.0–30.3. | Retain `1,4,4` for that configuration. Revisit for a different device balance or scheduling design. |
| R02 | Restrict pull-kernel CTA counts | Repeated settings spanning 12–82 CTAs were about 22.0–22.7 ms/token, within reported noise. | No established gain from this sweep. A different kernel or hardware balance is a new experiment. |
| R03 | Replacement shared/resident expert kernels | Exact outputs, but shared kernel 50 → 60 µs; resident experts 193 → 297 µs. | Rejected implementations. Revisit with a different kernel design, not the same substitutions. |
| R04 | Elastic expert cache using prefill buffers (`XS_ELASTIC`) | Added slots, but decode 37.9 / 37.6 versus 38.1 tokens/s. Miss rate 17.4% versus 17.2%. | Prefill-hot selection failed to improve decode locality. A smaller permanent workspace and refill driven by decode statistics are untested alternatives. |
| R05 | Schedule main-GPU pull after resident/shared kernels (`XS_PULL_AFTER`) | Baseline 37.87–38.26 tokens/s; changed schedule 36.83–37.20. | Faster isolated compute did not offset lost overlap. Revisit only with changed dependencies or transport. |
| R06 | Pre-gating prefetch in the pull stream (`XS_PREFETCH`) | Bitwise checks passed. One slot added about 2.4 ms/token; two added about 5.8 ms/token. | Disabled. A separate prefetch stream, readiness checks and double buffering would be a different implementation. |
| R07 | Dynamic context allocation **plus** packed FP4 KV | About 100 additional resident experts, without a demonstrated decode gain; fresh prefill was about 7–14% slower in the reported comparison. | Combined configuration disabled. This result does not isolate the cost of dynamic allocation from packing. See U01. |
| R08 | Current speculative verification path | Ordinary decode: 25.7 ms/token. Verification of 1 / 2 / 3 / 4 tokens: 48.6 / 98.1 / 120.0 / 149.6 ms. | Not competitive in that path. Revisit only after a faster short-batch verifier and acceptance measurements on the target workload. |

## Reasoning and quality experiments

| ID | Experiment | Recorded result | Decision and limitation |
|---|---|---|---|
| Q01 | Full held-out decode evaluation, eight sequences / 8184 positions | Baseline KL 0.2268, top-1 80.35%; earlier optimized engine KL 0.2288, top-1 80.19%. | Passed the historical acceptance thresholds. This does not certify every later flag combination. |
| Q02 | Uncertainty on short factual answers | On a mixed 240-question calibration set, answer entropy without reasoning had AUROC 0.93. After reasoning, answer entropy/logprob had AUROC 0.35–0.51; reasoning length had AUROC 0.80. | Optional signals only. Dataset-dependent calibration, not a general hallucination detector. |
| Q03 | Additional reasoning controls in an agent workflow | One factual question, three runs per variant: no overlay median 223 s; existing hesitation penalty 125 s; narrower word list 107 s with a 346 s outlier; penalty + budget 175 s; min-p 196 s. | No repeatable improvement over the existing policy was established. One question is not a broad quality evaluation. |
| Q04 | Apply hesitation penalty to difficult agent tasks | On the reported lab subset, 2/6 tasks solved versus 3/6 without the global penalty. | Do not generalize a factual-question optimization to every request. The APIs remain opt-in. |
| Q05 | Long-input retrieval checks | At 307,730 input tokens: 5/5 planted facts, an absent fact was rejected, prefill about 308 s, decode about 37.5 tokens/s. An earlier roughly 480K check also retrieved 5/5 facts. | Scenario checks only; the later long run used a combined experimental configuration. This is not a general long-context quality or throughput guarantee. |

## Open experiments: do not label these as failed

| ID | Question | Evidence available | What a useful next experiment must change |
|---|---|---|---|
| U01 | Does dynamic context allocation help without FP4 packing? | The completed comparison bundled these changes. | Compare dynamic-only, packed-only and both against one identical baseline. |
| U02 | Can decode-driven elastic residency help? | R04 failed with prefill-hot refill; extra slots alone were insufficient. | Measure decode miss rate and throughput with a different refill policy. |
| U03 | Can cache statistics survive restarts usefully? | `XS_SESSION_HEAT` stores statistics between sessions in the current process. | Persistent statistics and initial placement are a separate design; include cold-start traffic. |
| U04 | Would CUDA graphs or kernel fusion help? | Historical profiling perturbed timing substantially. The reported 91% used kernel time from a traced run and the step duration of an untraced run. | Measure comparable traces and a working implementation. The old ratio is not a utilization measurement or proof of a hardware ceiling. |
| U05 | Can a fast batched verifier make speculation useful? | R08 measures the current verifier, not every possible verifier or draft method. | Implement and time short batches first, then measure accepted length and total latency. HIP results do not establish CUDA results. |
| U06 | Are smaller indexer workspaces beneficial alone? | `XS_INDEX_ROWS` was included in a combined run; a memory reduction was reported. | Isolate workspace size and measure long-input indexing as well as prefill. |
| U07 | Does parallel decode improve the desired workload? | Route traces showed two different conversations sharing few experts; throughput estimates were extrapolated from those traces. | Benchmark a working batched decoder; report each request's latency and aggregate throughput separately. |

## Measurement rules for future entries

1. Use identical token inputs, explicit sampling and repeated baseline/candidate runs.
   Record cache warm-up and expert miss rate; a random corpus slice is not a controlled pair.
2. Separate fresh prefill, incremental prefill, decode and end-to-end latency.
   A speed increase in one phase does not imply an increase in another.
3. For stability, report window percentiles and slow cases. An average near
   38 tokens/s does not establish a minimum of 37 tokens/s.
4. Compare compatible numeric paths and run the appropriate held-out quality check.
   Token arrays alone missed the adjacent-user-message tokenization regression in A07.
5. Record negative and inconclusive outcomes. Add the changed assumption before
   reopening an entry; keep new measurements separate from historical summaries.
