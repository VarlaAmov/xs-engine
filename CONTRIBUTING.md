# Contributing

Read [the experiment journal](docs/EXPERIMENTS.md) before proposing performance work.
If an implementation has already failed, describe the changed assumption or design
that makes another experiment worthwhile.

Keep performance and correctness claims separate. Record the backend, model format,
input/output lengths, flags, cache warm-up, sampling, repetitions and baseline.
Compare identical token inputs; report decode, prefill and user-visible first-token
latency separately. Include tail behavior when claiming stable throughput.

For numeric changes, check the CPU reference or an appropriate held-out evaluation.
Passing a small API smoke test does not establish unchanged model quality.
Run GPU tests only on an explicitly reserved device. Ignored full-model acceptance
tests require `DSX_TEST_EXCLUSIVE=1` and the `DSX_TEST_*` model paths they document.

Add the result to the experiment journal, including negative results. Retain public
reproduction material where available. Keep local service configuration, personal
paths, private network addresses, credentials and conversation transcripts out of
commits and attachments. Preserve third-party license notices.

Use a development branch and open a pull request against `main`.
