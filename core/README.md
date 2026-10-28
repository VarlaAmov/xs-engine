# Shared core

Model configuration, GGUF/safetensors readers, quantized formats, CPU reference
numerics, sampling, conversation state and HTTP APIs shared by the GPU backends.

```bash
cargo test --locked -p dsx-core
```

The bundled web interface retains its [upstream attribution](webui/README.md).
