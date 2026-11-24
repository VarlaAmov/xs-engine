# Experiments and results

**Reviewed:** 2026-11-05 · **Status:** current

This journal summarizes historical engineering measurements. Deployment identifiers,
private paths, service configuration and conversation logs are intentionally absent.
The original raw runs are not included in this public snapshot; numerical results
below are historical summaries, not independently reproducible benchmark claims.

The main series used the CUDA backend with DeepSeek V4.1 Flash 1017 GSQ-RCO IQ2_XS,
hybrid CPU/GPU experts and a constrained GPU-resident expert cache. Results compare
variants within that series. They are not guarantees on other systems, model formats
or contexts. Input sizes are in tokens; K means approximately one thousand unless
an exact count is given.

## Mixed Unsloth expert formats, 2026-11-04

The previous tri-split path accepted IQ2_XXS, IQ2_XS, IQ2_S and IQ3_XXS only.
Unsloth UD-IQ1_M includes IQ1_S gate/up tensors, so one unsupported layer disabled
tri-split for the whole model. UD-IQ2_XXS also includes a Q2_K down tensor, whose
ggml type ID 10 was not recognized. This changes the format assumption of A01;
it does not repeat the split-ratio sweep in R01.

IQ1_S now supports strided CPU rows and CUDA pull kernels using exact integer
`8 * ternary_grid +/- 1` weights. Q2_K has a scalar decoder, CPU SIMD kernels,
raw and losslessly repacked CUDA GEMV, resident MoE, prefill and pull kernels.
The two Q2_K factors, `d` and `dmin`, remain independent; GGUF bits are retained.

Correctness results on CUDA 13 / sm86 / RTX 3090:

- The scalar Q2_K fixture matches an independently compiled ggml decoder hash;
  CPU SIMD and strided slices match the scalar path. Mixed CPU MoE slices match
  unsliced execution, including IQ1_S/Q2_K and Q2_K/IQ1_S.
- Resident GEMV, raw Q2_K GEMV and raw/repacked prefill checks pass at widths
  5120 and 2304 with 1, 3 and 6 activation vectors. Q2_K's maximum GEMV error
  divided by reference RMS is at most `4.1e-7` against the int8-activation reference.
- Pinned-arena pull tests cover seven format pairs, missed/resident picks, an
  empty miss set and 256/1024/1536-column slices with nonzero offsets. Maximum
  normalized down error is `6.81e-7`; gate/up differences after BF16 rounding
  stay below `6.66e-5`. Unassigned activation rows remain untouched.

The [raw kernel results](results/mixed-iq-correctness-20261104.txt) are correctness
checks, not performance measurements. Three mixed resident MoE checks also passed
in an exclusive environment. The HIP backend has not been qualified for these paths.

A subsequent exclusive full-model comparison on two RTX 3090s used identical UD-IQ1_M weights,
the same IQ4_NL Engram replacement, fixed expert placement (`DSX_ADAPT=0`),
arena 150 GB / Engram cache 6 GB and the first 1024-token held-out sequence with
top-256 teacher logits. The old build disabled tri-split; the candidate enabled it.
These runs used the corresponding deployment port, which also contains later
session and Engram updates absent from this public snapshot. They validate that
integration and are not an absolute throughput benchmark of this exact public tree.

| One sequence / 1023 decode positions | Previous build | Candidate |
|---|---:|---:|
| Teacher-forced decode time | 122.07 s | 48.59 s |
| Teacher-forced decode throughput | 8.38 tokens/s | 21.05 tokens/s |
| KL | 0.180392 | 0.180567 |
| Top-1 agreement | 84.8485% | 85.3372% |
| Perplexity | 9.7473 | 9.8000 |

This is one pair with background model download and checksum I/O still active.
Startup also differed: candidate host-memory allocation encountered reclamation
pressure. The decode times exclude startup. These numbers establish this run and
the working format path; they are not a general speedup percentage or production
throughput claim. The complete quality check is recorded below.

The eight-sequence held-out decode check subsequently passed (8184 positions):

| Same quality protocol | Previous historical run | Candidate |
|---|---:|---:|
| KL | 0.207267 | 0.207122 |
| Top-1 agreement | 81.2439% | 81.4272% |
| Perplexity | 8.9911 | 8.9931 |

The candidate took 384.06 s (21.31 teacher-forced tokens/s). The historical run
had different background I/O, so their throughput is not a controlled speed pair.
The full quality comparison passes the prior acceptance tolerance.

A generation smoke also passed with context allocation 327680, prefill chunk 4096,
8192 input tokens and 256 generated tokens: prefill 14.82 s (553 tokens/s), decode
10.42 s (24.56 tokens/s). All potentially missed experts had pinned-arena copies.
This is an 8K input with a 320K allocation; a complete 320K input, vision and
adaptive placement with these weights were not qualified by this run. Default
GPU headroom was retained; filling both devices with experts would displace
the memory needed by dense weights, KV and workspaces.
The [model summary](results/mixed-iq-model-summary-20261105.json) records the
protocol, measurements and limitations without deployment-specific inputs.

UD-IQ2_XXS then completed the same eight-sequence fixed-placement quality protocol:
KL 0.201833, top-1 agreement 81.8793%, perplexity 9.7510 and 18.50 teacher-forced
tokens/s. All 8184 positions completed with tri-split enabled and no experts left
to file fallback. It improved KL/top-1 over UD-IQ1_M but had worse perplexity on
this limited evaluation.

A final generation pair used the candidate build for both models, identical 8192
input IDs, 512 output tokens, context allocation 327680, chunk 4096, adaptive
placement (`DSX_ADAPT=16`, early adaptation), arena 150 GB / Engram cache 6 GB,
and no saved heat state:

| Adaptive generation pair | XS GSQ-RCO IQ2_XS | UD-IQ1_M |
|---|---:|---:|
| Prefill | 14.36 s / 571 tokens/s | 13.95 s / 587 tokens/s |
| Decode | 13.16 s / 38.92 tokens/s | 13.07 s / 39.16 tokens/s |

Thus the earlier IQ1_M slowdown was specific to fixed versus adaptive placement,
not reproduced under matching conditions. The 0.6% decode and 2.8% prefill
differences are single-run results, not a general IQ1_M performance advantage.
A complete 320K input remains unqualified. UD-IQ2_XXS with an adaptive main store
and static aux store reached 32.17–32.48 warm tokens/s, below UD-IQ1_M. Making
both XXS stores adaptive at a 150 GB arena was not accepted or qualified.

### Q6 dense optimization and model selection, 2026-11-05

The IQ1_M profile showed 3.73 ms/token in q and 4.46 ms/token in output projection.
A new 16-lane/four-row variant (`gemv_q6q16`) improved the model-shaped `o_b`
microbenchmark from 59.0 to 52.8 µs and the output head from 842.2 to 781.9 µs.
It was slower than `gemv_q6s8` for `q_b`, so `q_b` retains `s8`. Wider-row variants
were slower and removed. The BF16 rounding of Q6 `o_a` moved into the existing
`s8` epilogue, removing a separate kernel without changing that rounding step.

| Rewound warm decode, two passes | Previous candidate | Q16 + BF16 epilogue |
|---|---:|---:|
| 32K input | 39.61 / 40.22 tokens/s | 42.52 / 42.06 tokens/s |
| Real 94K agent turn | 39.13 / 38.55 tokens/s | 39.91 / 39.78 tokens/s |

This is about +6% at 32K and +2.6% for the 94K turn by the two-pass mean.
The q16 reduction changes FP32 accumulation order, so bitwise logits equality was
not expected. The full eight-sequence evaluation passed with KL 0.206163 versus
0.207122, top-1 agreement 81.3661% versus 81.4272%, perplexity 9.0432 versus
8.9931 and teacher-forced decode 22.34 versus 21.31 tokens/s. The small numerical
drift is accepted within the existing quality tolerance; it is not a claim of
universally better perplexity.

UD-IQ1_M is selected over UD-IQ2_XXS for this configuration: substantially higher
throughput and lower perplexity outweigh XXS's better KL/top-1 on this limited
evaluation. The optimized server also passed arithmetic API, three vision cases,
8K/32K streaming and a clean restart that loaded saved expert heat. A complete
320K input remains untested. The [model summary](results/mixed-iq-model-summary-20261105.json)
includes these results.

Format support does not qualify every residency policy. Tri-split requires pinned
copies of every expert that an adaptive store may evict; static stores need no
host duplicates. If the selected arena cannot hold all possible misses, the
current implementation disables tri-split rather than mixing incompatible CPU
slice rules.

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
| A09 | Q6 `q16` on large long-K matrices and BF16 `o_a` epilogue | About +6% warm decode at 32K and +2.6% on a real 94K turn; full quality eval passed. | Retained for UD-IQ1_M. `q_b` stays on `s8`; q16 changes FP32 accumulation order and requires quality qualification. |

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
| R09 | IQ1_M cache and placement sweeps | Replacement limits 24/32 and pull CTA changes were noise; alternate splits reached 39.1–39.8 tokens/s, static aux 39.25, CPU24 40.46 and reduced headroom 40.85–41.08 versus about 41–42 for the baseline. | Retain split `1,4,4`, 18 CPU workers, adaptive aux and default reserves for this workload. Do not repeat without a changed bottleneck. |

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
