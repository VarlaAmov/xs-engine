# Encoding fixtures

Official inputs and reference outputs originate from
[DeepSeek's encoding implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/main/encoding).
The original fixtures are preserved. Additional cases exercise tool calls, thinking
settings, message histories and compatibility adaptations.

`expectations.json` and `*.server.txt` retain known renderer differences separately
from the reference output. Passing such a test does not imply equality with the
reference encoder. The tokenizer's [license](../LICENSE-tokenizer) is included.

```bash
python3 -I core/tests/data/dsv41_encoding/gen_extra.py ./reference/encoding
cargo test --locked -p dsx-core --test encoding_golden
```
