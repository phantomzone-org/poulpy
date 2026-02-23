# Poulpy — Claude Code Instructions

---

## Project Summary

Poulpy is a Rust workspace implementing RLWE-based FHE lattice cryptography primitives.
It is structured as a hardware abstraction layer (HAL) + multiple CPU backends.

```
poulpy/
├── poulpy-hal/          # HAL: Backend/OEP traits + reference implementations
│   └── src/
│       ├── api/         # Public API traits (ModuleN, ModuleNew, TakeSlice, …)
│       ├── layouts/     # Core types (Module, VecZnx, VecZnxDft, VmpPMat, …)
│       ├── oep/         # Open Extension Point unsafe traits (*Impl)
│       ├── test_suite/  # Cross-backend test helpers (test_vec_znx_dft_add, …)
│       └── reference/
│           ├── fft64/   # f64 FFT backend (reim/, reim4/)
│           ├── ntt120/  # Q120 NTT backend — primitive + composite fns (COMPLETE)
│           └── vec_znx/ # ZnX coefficient-domain operations
├── poulpy-core/         # Backend-agnostic RLWE FHE primitives (~101 files)
├── poulpy-cpu-ref/      # Reference CPU backend: FFT64Ref (f64, scalar)
├── poulpy-cpu-avx/      # AVX2+FMA backend: FFT64AVX
├── poulpy-cpu-ntt120/   # NEW NTT120 CPU backend: NTT120Ref (COMPLETE, conv stub)
└── poulpy-schemes/      # Higher-level FHE scheme implementations
```

---

## Tech Stack

- **Language**: Rust (edition 2024), workspace with `resolver = "3"`
- **Key deps**: `bytemuck`, `rand`/`rand_distr`/`rand_chacha`, `once_cell`, `itertools`
- `#![deny(rustdoc::broken_intra_doc_links)]` is active in poulpy-hal

---

## Backend Architecture

Every backend is a zero-sized marker struct implementing `Backend`:

```rust
pub trait Backend: Sized + Sync + Send {
    type ScalarBig:  Copy + Zero + Display + Debug + Pod;  // big-coeff repr
    type ScalarPrep: Copy + Zero + Display + Debug + Pod;  // DFT/NTT domain
    type Handle: 'static;                                  // precomputed tables
    fn layout_prep_word_count() -> usize;
    fn layout_big_word_count()  -> usize;
    unsafe fn destroy(handle: NonNull<Self::Handle>);
}
```

| Backend       | ScalarPrep     | ScalarBig | prep_word | big_word |
|---------------|----------------|-----------|-----------|----------|
| `FFT64Ref`    | `f64`          | `i64`     | 2         | 1        |
| `FFT64AVX`    | `f64`          | `i64`     | 2         | 1        |
| `NTT120Ref`   | `Q120bScalar`  | `i128`    | 1         | 1        |

`Q120bScalar = [u64; 4]` (32 bytes) — four CRT residues per NTT coefficient.

Operations are implemented via **OEP (Open Extension Point)** `unsafe trait *Impl`
in `poulpy_hal::oep`, delegating to reference functions in `poulpy_hal::reference::`.

---

## NTT120 Backend Status

See [`docs/ntt120-backend.md`](docs/ntt120-backend.md) for design details.

### `poulpy-hal/src/reference/ntt120/` — COMPLETE
`primes.rs`, `types.rs`, `arithmetic.rs`, `mat_vec.rs`, `ntt.rs`,
`vec_znx_big.rs`, `vec_znx_dft.rs`, `svp.rs`, `vmp.rs`

### `poulpy-cpu-ntt120/src/` — COMPLETE (convolution is a runtime stub)
`lib.rs`, `module.rs`, `scratch.rs`, `znx.rs`, `vec_znx.rs`, `vec_znx_big.rs`,
`vec_znx_dft.rs`, `svp.rs`, `vmp.rs`, `convolution.rs` (panics with `unimplemented!()`)

### Next TODOs for NTT120
1. **Write tests** in `poulpy-cpu-ntt120/src/tests.rs` using `poulpy_hal::test_suite`:
   - `test_suite::vec_znx_dft`: `test_vec_znx_dft_add`, `test_vec_znx_idft_apply`,
     `test_vec_znx_idft_apply_consume`, `test_vec_znx_dft_sub`, …
   - `test_suite::svp`: `test_svp_apply_dft`, `test_svp_apply_dft_to_dft`,
     `test_svp_apply_dft_to_dft_add`, `test_svp_apply_dft_to_dft_inplace`
   - `test_suite::vmp`: `test_vmp_apply_dft`, `test_vmp_apply_dft_to_dft`,
     `test_vmp_apply_dft_to_dft_add`
   - All helpers take `(base2k: usize, module_ref: &Module<FFT64Ref>, module_test: &Module<NTT120Ref>)`
2. **Implement convolution** — port FFT64 convolution to NTT120 q120b/q120c arithmetic.
3. **Wire NTT120Ref into poulpy-core/poulpy-schemes** as needed.

---

## Verification Commands

```bash
cargo check -p poulpy-hal
cargo check -p poulpy-cpu-ntt120
cargo doc -p poulpy-hal --no-deps
cargo doc -p poulpy-cpu-ntt120 --no-deps
cargo test -p poulpy-hal ntt120        # 2 tests pass
cargo check -p poulpy-core
cargo check -p poulpy-cpu-avx
```

All of the above currently pass with zero warnings/errors.

---

## spqlios-arithmetic Port Policy

Files ported from [spqlios-arithmetic](https://github.com/tfhe/spqlios-arithmetic) must start
with the standard DISCLAIMER block. Example:

```rust
// ----------------------------------------------------------------------
// DISCLAIMER
//
// This module contains code that has been directly ported from the
// spqlios-arithmetic library
// (https://github.com/tfhe/spqlios-arithmetic), which is licensed
// under the Apache License, Version 2.0.
//
// The porting process from C to Rust was done with minimal changes
// in order to preserve the semantics and performance characteristics
// of the original implementation.
//
// Both Poulpy and spqlios-arithmetic are distributed under the terms
// of the Apache License, Version 2.0. See the LICENSE file for details.
//
// ----------------------------------------------------------------------
```

Files that do NOT need the disclaimer: `mod.rs`, trivial wrappers, poulpy-specific code.
All `ntt120/` and ported `fft64/`/`cpu-avx/` files already carry the disclaimer.

---

## Common Pitfalls & Key Design Decisions

### NttModuleHandle / NttHandleProvider (orphan rule)
`NttModuleHandle` and `Module` are both in `poulpy-hal`, so backend crates **cannot**
write `impl NttModuleHandle for Module<TheirBackend>` (orphan rule).

**Solution**: backend crates implement `NttHandleProvider` for their local handle type:
```rust
// poulpy-hal: blanket impl
impl<B: Backend> NttModuleHandle for Module<B> where B::Handle: NttHandleProvider { … }

// poulpy-cpu-ntt120/module.rs: local-type impl (no orphan issue)
unsafe impl NttHandleProvider for NTT120RefHandle { … }
```

### PrimeSet must be in scope
`Primes30::Q` and `Primes30::CRT_CST` are associated items of the `PrimeSet` trait.
Always import: `use poulpy_hal::reference::ntt120::primes::{PrimeSet, Primes30};`

### Rust 2024: `unsafe_op_in_unsafe_fn`
Inside an `unsafe fn`, each unsafe operation still needs its own `unsafe {}` block.

### Intra-doc links in ntt120 submodules
Sibling items need `super::`: `[super::mat_vec::fn_name]`, not `[mat_vec::fn_name]`.

### Bulk File Edits
Edit tool "file has been read" check doesn't persist across messages.
Use a Python script via Bash for multi-file header prepending:
```bash
python3 - <<'EOF'
files = [...]
disclaimer = "..."
for f in files:
    with open(f) as fh: content = fh.read()
    if "DISCLAIMER" in content: continue
    with open(f, "w") as fh: fh.write(disclaimer + content)
EOF
```
