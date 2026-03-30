# poulpy-bench

Consolidated [Criterion](https://bheisler.github.io/criterion.rs/book/) benchmark suite for the poulpy workspace.

Each benchmark binary covers one subsystem. Binaries that operate on generic polynomial or FHE operations run against **all available backends**; binaries that target transform-domain-specific operations (DFT, convolution, VMP/SVP) are restricted to the **FFT64 family**.

## Backends and feature flags

| Backend | Type | Feature flag | Label in output |
|---|---|---|---|
| `FFT64Ref` | FFT64, portable | *(always enabled)* | `fft64-ref` |
| `NTT120Ref` | NTT120, portable | *(always enabled)* | `ntt120-ref` |
| `FFT64Avx` | FFT64, AVX2/FMA | `enable-avx` | `fft64-avx` |
| `NTT120Avx` | NTT120, AVX2/FMA | `enable-avx` | `ntt120-avx` |

The `enable-avx` flag enables the `poulpy-cpu-avx` backend and requires `target_arch = "x86_64"`.

## Benchmark binaries

### HAL-level (backend-agnostic polynomial arithmetic)

| Binary | Subsystem | Backends |
|---|---|---|
| `vec_znx` | `VecZnx` add / sub / negate / rotate / automorphism / shift | all |
| `vec_znx_big` | `VecZnxBig` add / sub / negate / normalize / automorphism | all |
| `vec_znx_dft` | DFT-domain add / sub / apply / iDFT | FFT64 only |
| `convolution` | Polynomial convolution (prepare + apply) | FFT64 only |
| `svp` | Scalar-vector product (prepare, DFT-to-DFT) | FFT64 only |
| `vmp` | Vector-matrix product (prepare, DFT-to-DFT) | FFT64 only |
| `fft` | Raw FFT / iFFT primitive | hardcoded (ref + avx) |
| `ntt` | Raw NTT / iNTT primitive | hardcoded (ref + avx) |

### Core-level (scheme-agnostic FHE operations)

| Binary | Subsystem | Backends |
|---|---|---|
| `operations` | GLWE add / sub / normalize / mul-plain | all |
| `encryption` | GLWE / GGSW / automorphism-key encryption | all |
| `decryption` | GLWE decryption | all |
| `automorphism` | GLWE automorphism | all |
| `external_product` | GLWE external product (inplace + out-of-place) | all |
| `keyswitch` | GLWE key-switching | all |

### Scheme-level (end-to-end FHE primitives)

| Binary | Subsystem | Backends |
|---|---|---|
| `blind_rotate` | Blind rotation (CGGI / AP) | hardcoded |
| `circuit_bootstrapping` | Circuit bootstrapping | hardcoded |
| `bdd_prepare` | BDD key preparation | hardcoded |
| `bdd_arithmetic` | BDD homomorphic arithmetic | hardcoded |

### Regression tracking

| Binary | Purpose |
|---|---|
| `standard` | One representative run across **all layers** with fixed parameters — used for version-to-version regression tracking |

The `standard` binary uses a single parameter set (`N=4096`, `base2k=18`, `k=54`, `rank=1`) for all HAL and core benchmarks, and the standard parameter sets embedded in `poulpy-schemes` for the scheme-level benchmarks. Its results can be saved as named baselines for direct comparison across releases (see [Save and compare baselines](#save-and-compare-baselines)).

## Running benchmarks

### All benchmarks, reference backends only

```sh
cargo bench -p poulpy-bench
```

### All benchmarks with AVX acceleration

```sh
cargo bench -p poulpy-bench --features enable-avx
```

### One binary

```sh
# run only the vec_znx binary
cargo bench -p poulpy-bench --bench vec_znx

# with AVX
cargo bench -p poulpy-bench --bench vec_znx --features enable-avx
```

### One group or function within a binary

Criterion accepts a regex filter as a trailing argument (after `--`).
The benchmark ID format is `<group_name>/<parameter>` where the group name
encodes the operation and backend label.

```sh
# all vec_znx benchmarks on the ntt120-ref backend
cargo bench -p poulpy-bench --bench vec_znx -- ntt120-ref

# only the add benchmark, all backends
cargo bench -p poulpy-bench --bench vec_znx -- vec_znx_add

# one specific backend × operation
cargo bench -p poulpy-bench --bench vec_znx -- "vec_znx_add::fft64-ref"

# all encryption benchmarks, AVX only
cargo bench -p poulpy-bench --bench encryption --features enable-avx -- avx
```

### Standard regression binary

```sh
# run the standard binary (ref backends)
cargo bench -p poulpy-bench --bench standard

# with AVX acceleration
cargo bench -p poulpy-bench --bench standard --features enable-avx
```

### Save and compare baselines

```sh
# save a named baseline (e.g. tagging a release)
cargo bench -p poulpy-bench --bench standard -- --save-baseline v0.4.4

# run again later and compare against it
cargo bench -p poulpy-bench --bench standard -- --baseline v0.4.4
```

The same `--save-baseline` / `--baseline` flags work on any bench binary:

```sh
# save a baseline named "before"
cargo bench -p poulpy-bench --bench vec_znx -- --save-baseline before

# bench again and compare
cargo bench -p poulpy-bench --bench vec_znx -- --baseline before
```

Criterion HTML reports are written to `target/criterion/`.

## Adding a new backend

1. Add the backend crate to `[dependencies]` in `Cargo.toml` (behind an optional feature if needed).
2. Add one entry to the appropriate private family macro in [`src/lib.rs`](src/lib.rs):
   - `for_each_fft_backend_family!` for an FFT64 backend
   - `for_each_ntt_backend_family!` for an NTT120 backend
3. No bench files need to change.
