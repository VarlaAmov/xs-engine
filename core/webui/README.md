# Embedded web interface

The interface served by `dsx-server` derives from
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s web UI.
Its upstream [MIT license](LICENSE-llama.cpp) is preserved.
`dist/` is embedded into the server at build time; `build.sh` records the pinned
source revision and rebuild procedure. The engine-specific logo is applied there.
