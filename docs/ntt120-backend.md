# NTT120 Backend — Design Notes

## Overview

The NTT120 backend uses four ~30-bit primes (Primes30, Q ≈ 2^120) and Chinese
Remainder Theorem (CRT) to perform exact polynomial multiplication without
floating-point error. It replaces the f64 FFT backend for use cases that require
exact arithmetic at larger moduli.

## Prime Set

`Primes30`: four primes, each ~30 bits, product Q ≈ 2^120.
- Accessed via the `PrimeSet` trait: `Primes30::Q: [u32; 4]`, `Primes30::CRT_CST: [u32; 4]`
- **`PrimeSet` must be in scope** to use associated constants on `Primes30`.

## Memory Layouts

| Poulpy type | Bytes/coeff | Format |
|-------------|-------------|--------|
| `VecZnx`    | 8 (i64)     | coefficient domain |
| `VecZnxBig` | 16 (i128)   | CRT-reconstructed exact value |
| `VecZnxDft` | 32 (Q120bScalar = 4×u64) | NTT domain, one u64 per prime |
| `SvpPPol`   | 32 (Q120cScalar = 8×u32) | NTT+Montgomery domain |
| `VmpPMat`   | 32 (Q120cScalar)          | same as SvpPPol |

## HAL Reference Files (`poulpy-hal/src/reference/ntt120/`)

### `primes.rs`
- `PrimeSet` trait: associated consts `Q`, `CRT_CST`, `N_PRIMES`
- `Primes29`, `Primes30`, `Primes31` structs

### `types.rs`
- `Q120aScalar`, `Q120bScalar`, `Q120cScalar` newtype wrappers
- `Q120x2bScalar`, `Q120x2cScalar` paired variants

### `arithmetic.rs`
- `b_from_znx64_ref`: i64 poly → q120b (4 u64 CRT residues per coeff)
- `c_from_znx64_ref`: i64 poly → q120c (Montgomery form)
- `b_to_znx128_ref`: q120b → i128 (CRT reconstruction)
- `add_bbb_ref`: q120b + q120b → q120b
- `c_from_b_ref`: q120b → q120c

### `mat_vec.rs`
- `BbbMeta<P>`, `BbcMeta<P>`: precomputed lazy-reduction metadata
- `vec_mat1col_product_bbc_ref`: q120c × q120b → q120b (1 column)
- `vec_mat1col_product_x2_bbc_ref`: 2-coefficient block version
- `vec_mat2cols_product_x2_bbc_ref`: 2-column × 2-coeff block

### `ntt.rs`
- `NttTable<P>`, `NttTableInv<P>`: precomputed twiddle tables
- `ntt_ref<P>`: in-place forward NTT (modifies slice)
- `intt_ref<P>`: in-place inverse NTT (modifies slice)

### `vec_znx_dft.rs`
- `NttModuleHandle` trait: `get_ntt_table`, `get_intt_table`, `get_bbc_meta`
- `NttHandleProvider` unsafe trait: implemented by backend handle types
- Blanket: `impl<B: Backend> NttModuleHandle for Module<B> where B::Handle: NttHandleProvider`
- `ntt120_vec_znx_dft_apply`: i64 VecZnx → q120b VecZnxDft (forward NTT)
- `ntt120_vec_znx_idft_apply_tmp_bytes(n) = 4*n*8`
- `ntt120_vec_znx_idft_apply`: q120b → i128 (non-destructive, uses scratch)
- `ntt120_vec_znx_idft_apply_tmpa`: q120b → i128 (uses input as scratch)
- `ntt120_vec_znx_dft_{add,sub,add_inplace,sub_inplace,sub_negate_inplace,add_scaled_inplace,copy,zero}`

### `svp.rs`
- `ntt120_svp_prepare`: i64 scalar poly → q120c SvpPPol
- `ntt120_svp_apply_dft_to_dft`: q120c × q120b → q120b (per-coeff, calls bbc)
- `ntt120_svp_apply_dft_to_dft_add`: accumulate variant
- `ntt120_svp_apply_dft_to_dft_inplace`: in-place variant

### `vmp.rs`
- `ntt120_vmp_prepare_tmp_bytes(n) = 4*n*8`
- `ntt120_vmp_prepare`: i64 MatZnx rows → q120c VmpPMat
- `ntt120_vmp_apply_dft_to_dft_tmp_bytes(a_size, b_rows, b_cols_in)`
- `ntt120_vmp_apply_dft_to_dft`: q120b vector × q120c matrix → q120b
- `ntt120_vmp_apply_dft_to_dft_add`: accumulate variant
- `ntt120_vmp_zero`: zero a VmpPMat

## Backend Crate (`poulpy-cpu-ntt120/src/`)

### `module.rs`
- `NTT120RefHandle { table_ntt, table_intt, meta_bbc }`
- `unsafe impl NttHandleProvider for NTT120RefHandle` (wires into blanket impl)
- `impl Backend for NTT120Ref` with `ScalarPrep = Q120bScalar`, `ScalarBig = i128`

### `vec_znx_dft.rs` — Key: `compact_all_blocks`
IDFT-consume must convert in-place from Q120b layout (32B/coeff) to i128 (16B/coeff).
Processing blocks in order k=0,1,...,n_blocks-1 is safe because:
- For k≥1: dst end = 16n(k+1) ≤ src start = 32nk → non-overlapping
- For k=0: all four u64 residues are read into locals before any i128 write

### `convolution.rs`
Runtime stub — all methods panic with `unimplemented!()`. Future work.

## Test Suite Available

`poulpy_hal::test_suite` provides cross-backend correctness tests.
All helpers compare `Module<FFT64Ref>` (reference) vs `Module<NTT120Ref>` (test).
Signature: `test_foo(base2k: usize, module_ref: &Module<FFT64Ref>, module_test: &Module<NTT120Ref>)`

Available:
- `vec_znx_dft`: add, add_inplace, sub, sub_inplace, sub_negate_inplace, copy,
  idft_apply, idft_apply_tmpa, idft_apply_consume
- `svp`: apply_dft, apply_dft_to_dft, apply_dft_to_dft_add, apply_dft_to_dft_inplace
- `vmp`: apply_dft, apply_dft_to_dft, apply_dft_to_dft_add
- `vec_znx`: (various coefficient-domain tests)
- `vec_znx_big`: (large-coefficient tests)
- `convolution`: test_convolution, test_convolution_by_const, test_convolution_pairwise

See `poulpy-cpu-ref/src/tests.rs` for usage examples.
