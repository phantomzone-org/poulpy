# Poulpy-CPU-IFMA

**Poulpy-CPU-IFMA** is a Rust crate that provides an **AVX512-IFMA accelerated CPU backend for Poulpy**.

This backend implements the Poulpy HAL unified `HalImpl` trait and can be used by:

- [`poulpy-hal`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-hal)
- [`poulpy-core`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-core)
- [`poulpy-bin-fhe`, `poulpy-ckks`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-schemes)

## Safety and Requirements

To avoid illegal hardware instructions (SIGILL) on unsupported CPUs, this backend is **opt-in** and **only builds when explicitly requested**.

| Requirement | Status |
|------------|--------|
| Cargo feature flag | `--features enable-ifma` **must be enabled** |
| CPU architecture | `x86_64` |
| CPU target features | `AVX512F` + `AVX512IFMA` + `AVX512VL` |

If `enable-ifma` is enabled but the target does not provide these capabilities, the build **fails immediately with a clear error message**, rather than generating invalid binaries.

When `enable-ifma` is **not** enabled, this crate is simply skipped and Poulpy automatically falls back to the portable `poulpy-cpu-ref` backend. This keeps the workspace portable on unsupported targets.

## Building with the IFMA backend enabled

Because the compiler must generate AVX-512 IFMA instructions, both the Cargo feature and CPU target flags must be specified:

```bash
RUSTFLAGS="-C target-feature=+avx512f,+avx512ifma,+avx512vl" \
cargo build --features enable-ifma
```

On a host that supports these instructions natively, `target-cpu=native` also works:

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

This crate exposes one AVX512-IFMA backend:

```rust
use poulpy_cpu_ifma::NTTIfma;
use poulpy_hal::{api::ModuleNew, layouts::Module};

let log_n: usize = 10;

// Q120 NTT backend (AVX512-IFMA, CRT over three ~40-bit primes)
let module: Module<NTTIfma> = Module::<NTTIfma>::new(1 << log_n);
```

Once compiled with `enable-ifma`, the backend can be used transparently anywhere Poulpy expects a backend type (`poulpy-hal`, `poulpy-core`, `poulpy-bin-fhe`, `poulpy-ckks`).

## Contributors

To implement your own Poulpy backend (SIMD or accelerator):

1. Define a backend struct and implement the `Backend` trait
2. Implement the unified `HalImpl` trait (via macros delegating to `hal_defaults`, or manually)
3. Implement the `CoreImpl` trait from `poulpy-core` (via the `impl_core_default_methods!` macro)

Your backend will automatically integrate with `poulpy-hal`, `poulpy-core`, `poulpy-bin-fhe`, and `poulpy-ckks`. No modifications to those crates are required.
