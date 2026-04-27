# 🐙 Poulpy-CPU-AVX512

**Poulpy-CPU-AVX512** is a Rust crate that provides **AVX-512 accelerated CPU backends for Poulpy**.

It exposes two backends, gated behind two distinct Cargo features:

- **`FFT64Avx512`** — f64 complex-FFT backend, gated on `enable-avx512f` (requires AVX-512F only).
- **`NTT120Ifma`** — Q120 NTT backend (CRT over three ~40-bit primes), gated on `enable-ifma` (requires AVX-512F + AVX-512-IFMA + AVX-512VL).

`enable-ifma` implies `enable-avx512f`, so enabling IFMA builds both backends.

This crate implements the Poulpy HAL extension traits and can be used by:

- [`poulpy-hal`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-hal)
- [`poulpy-core`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-core)
- [`poulpy-schemes` (CKKS, bin-FHE)](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-schemes)

## 🚩 Safety and Requirements

To avoid illegal hardware instructions (SIGILL) on unsupported CPUs, the backends are **opt-in** and **only build when explicitly requested**.

| Feature | CPU target features required |
|---------|------------------------------|
| `enable-avx512f` (builds `FFT64Avx512`) | `AVX512F` |
| `enable-ifma` (additionally builds `NTT120Ifma`) | `AVX512F` + `AVX512IFMA` + `AVX512VL` |

If a feature is enabled but the target does not provide the required capabilities, the build **fails immediately with a clear error message**, rather than generating invalid binaries.

When neither feature is enabled, this crate is simply skipped and Poulpy automatically falls back to the portable `poulpy-cpu-ref` backend. This ensures that Poulpy's workspace remains portable (e.g. for macOS ARM).

## ⚙️ Building

For the AVX-512F-only `FFT64Avx512` backend:

```bash
RUSTFLAGS="-C target-feature=+avx512f" \
cargo build --features enable-avx512f
```

For both backends (AVX-512F + IFMA):

```bash
RUSTFLAGS="-C target-feature=+avx512f,+avx512ifma,+avx512vl" \
cargo build --features enable-ifma
```

On a host that natively supports the required instructions, `target-cpu=native` also works:

```bash
RUSTFLAGS="-C target-cpu=native" \
cargo build --features enable-ifma
```

### Running an example

```bash
RUSTFLAGS="-C target-feature=+avx512f,+avx512ifma,+avx512vl" \
cargo run --example <name> --features enable-ifma
```

### Running benchmarks

```bash
RUSTFLAGS="-C target-feature=+avx512f,+avx512ifma,+avx512vl" \
cargo bench --features enable-ifma
```

## Basic Usage

```rust
use poulpy_cpu_avx512::{FFT64Avx512, NTT120Ifma};
use poulpy_hal::{api::ModuleNew, layouts::Module};

let log_n: usize = 10;

// f64 FFT backend (AVX-512F)
let module: Module<FFT64Avx512> = Module::<FFT64Avx512>::new(1 << log_n);

// Q120 NTT backend (AVX-512-IFMA, CRT over three ~40-bit primes)
let module: Module<NTT120Ifma> = Module::<NTT120Ifma>::new(1 << log_n);
```

Each backend is usable transparently anywhere Poulpy expects a backend type (`poulpy-hal`, `poulpy-core`, `poulpy-schemes`).

## 🤝 Contributors

To implement your own Poulpy backend (SIMD or accelerator):

1. Define a backend struct
2. Implement the open extension traits from `poulpy-hal/oep`
3. Implement the `Backend` trait

Your backend will automatically integrate with:

* `poulpy-hal`
* `poulpy-core`
* `poulpy-schemes` (CKKS, bin-FHE)

No modifications to those crates are required — the HAL provides the extension points.

---

For questions or guidance, feel free to open an issue or discussion in the repository.
