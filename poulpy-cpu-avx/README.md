# 🐙 Poulpy-CPU-AVX

**Poulpy-CPU-AVX** is a Rust crate that provides an **AVX2 + FMA accelerated CPU backend for Poulpy**.

This backend implements the Poulpy HAL extension traits and can be used by:

- [`poulpy-hal`](https://github.com/phantomzone-org/poulpy/tree/main/poulpy-hal)
- [`poulpy-core`](https://github.com/phantomzone-org/poulpy/tree/main/poulpy-core)
- [`poulpy-schemes`](https://github.com/phantomzone-org/poulpy/tree/main/poulpy-schemes)

## 🚩 Safety and Requirements

To avoid illegal hardware instructions (SIGILL) on unsupported CPUs, this backend is **opt-in** and **only builds when explicitly requested**.

| Requirement | Status |
|------------|--------|
| Cargo feature flag | `--features enable-avx` **must be enabled** |
| CPU architecture | `x86_64` |
| CPU target features | `AVX2` + `FMA` |

If `enable-avx` is enabled but the target does not provide these capabilities, the build **fails immediately with a clear error message**, rather than generating invalid binaries.

When `enable-avx` is **not** enabled, this crate is simply skipped and Poulpy automatically falls back to the portable `poulpy-cpu-ref` backend. This ensure that Poulpy's workspace remains portable (e.g. for macOS ARM).

## ⚙️ Building with the AVX backend enabled

Because the compiler must generate AVX2 + FMA instructions, both the Cargo feature and CPU target flags must be specified:

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" \
cargo build --features enable-avx
````

### Running an example

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" \
cargo run --example <name> --features enable-avx
```

### Running benchmarks

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" \
cargo bench --features enable-avx
```

## Basic Usage

This crate exposes two AVX2-accelerated backends:

```rust
use poulpy_cpu_avx::{FFT64Avx, NTT120Avx};
use poulpy_hal::{api::ModuleNew, layouts::Module};

let log_n: usize = 10;

// f64 FFT backend (AVX2 + FMA)
let module: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << log_n);

// Q120 NTT backend (AVX2, CRT over four ~30-bit primes)
let module: Module<NTT120Avx> = Module::<NTT120Avx>::new(1 << log_n);
```

Once compiled with `enable-avx`, both backends are usable transparently anywhere Poulpy expects a backend type (`poulpy-hal`, `poulpy-core`, `poulpy-schemes`).

## 🤝 Contributors

To implement your own Poulpy backend (SIMD or accelerator):

1. Define a backend struct
2. Implement the open extension traits from `poulpy-hal/oep`
3. Implement the `Backend` trait

Your backend will automatically integrate with:

* `poulpy-hal`
* `poulpy-core`
* `poulpy-schemes`

No modifications to those crates are required — the HAL provides the extension points.

---

For questions or guidance, feel free to open an issue or discussion in the repository.
