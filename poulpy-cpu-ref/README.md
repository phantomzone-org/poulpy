# 🐙 Poulpy-CPU-REF

**Poulpy-CPU-REF** is the **reference (portable) CPU backend for Poulpy**.

It implements the Poulpy HAL extension traits without requiring SIMD or specialized CPU instructions, making it suitable for:

- all CPU architectures (`x86_64`, `aarch64`, `arm`, `riscv64`, …)
- development machines and CI runners
- environments without AVX or other advanced SIMD support

This backend integrates transparently with:

- `poulpy-hal`
- `poulpy-core`
- `poulpy-schemes`

---

## When is this backend used?

`poulpy-cpu-ref` is always available and requires **no compilation flags and no CPU features**.

It is automatically selected when:

- the project does not request an optimized backend, or
- the target CPU does not support the requested SIMD backend (e.g., AVX), or
- portability and reproducibility are more important than raw performance.

No additional configuration is required to use it.

---

## 🧪 Basic Usage

```rust
use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::{api::ModuleNew, layouts::Module};

let log_n: usize = 10;
let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << log_n);
```

This works on **all supported platforms and architectures**.

---

## Performance Notes

`poulpy-cpu-ref` prioritizes:

* portability
* correctness
* ease of debugging

For maximum performance on x86_64 CPUs with AVX2 + FMA support, consider enabling the optional optimized backend:

```
poulpy-cpu-avx (feature: enable-avx)
```

Benchmarks and applications can freely switch between backends without changing source code — backend selection can be handled with feature flags, for example

```rust
#[cfg(all(feature = "enable-avx", target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
use poulpy_cpu_avx::FFT64Avx as BackendImpl;

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
use poulpy_cpu_ref::FFT64Ref as BackendImpl;
```

---

## 🤝 Contributors

To implement your own backend (SIMD or accelerator):

1. Define a backend struct
2. Implement the open extension traits from `poulpy-hal/oep`
3. Implement the `Backend` trait

Your backend will automatically integrate with:

* `poulpy-hal`
* `poulpy-core`
* `poulpy-schemes`

No modifications to those crates are necessary — the HAL provides the extension points.

---

For questions or guidance, feel free to open an issue or discussion in the repository.

```
