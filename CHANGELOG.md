# CHANGELOG

## [0.4.1] - 2025-11-26

### Summary
- Update convolution API to match spqlios-arithmetic & removed API for bivariate tensoring.

## `poulpy-hal`
- Removed `Backend` generic from `VecZnxBigAllocBytesImpl`.
- Add `CnvPVecL` and `CnvPVecR` structs.
- Add `CnvPVecBytesOf` and `CnvPVecAlloc` traits.
- Add `Convolution` trait, which regroups the following methods:
  - `cnv_prepare_left_tmp_bytes`
  - `cnv_prepare_left`
  - `cnv_prepare_right_tmp_bytes`
  - `cnv_prepare_right`
  - `cnv_by_const_apply`
  - `cnv_by_const_apply_tmp_bytes`
  - `cnv_apply_dft_tmp_bytes`
  - `cnv_apply_dft`
  - `cnv_pairwise_apply_dft_tmp_bytes`
  - `cnv_pairwise_apply_dft`
- Add the following Reim4 traits:
  - `Reim4Convolution`
  - `Reim4Convolution1Coeff`
  - `Reim4Convolution2Coeffs`
  - `Reim4Save1BlkContiguous`
- Add the following traits:
  - `i64Save1BlkContiguous`
  - `i64Extract1BlkContiguous`
  - `i64ConvolutionByConst1Coeff`
  - `i64ConvolutionByConst2Coeffs`
- Update signature `Reim4Extract1Blk` to `Reim4Extract1BlkContiguous`.
- Add fft64 backend reference code for 
  - `reim4_save_1blk_to_reim_contiguous_ref`
  - `reim4_convolution_1coeff_ref`
  - `reim4_convolution_2coeffs_ref`
  - `convolution_prepare_left`
  - `convolution_prepare_right`
  - `convolution_apply_dft_tmp_bytes`
  - `convolution_apply_dft`
  - `convolution_pairwise_apply_dft_tmp_bytes`
  - `convolution_pairwise_apply_dft`
  - `convolution_by_const_apply_tmp_bytes`
  - `convolution_by_const_apply`
- Add `take_cnv_pvec_left` and `take_cnv_pvec_right` methods to `ScratchTakeBasic` trait.
- Add the following tests methods for convolution:
  - `test_convolution`
  - `test_convolution_by_const`
  - `test_convolution_pairwise`
- Add the following benches methods for convolution:
  - `bench_cnv_prepare_left`
  - `bench_cnv_prepare_right`
  - `bench_cnv_apply_dft`
  - `bench_cnv_pairwise_apply_dft`
  - `bench_cnv_by_const`
- Update normalization API and OEP to take `res_offset: i64`. This allows the user to specify a bit-shift (positive or negative) applied to the normalization. Behavior-wise, the bit-shift is applied before the normalization (i.e. before applying mod 1 reduction). Since this is an API break, opportunity was taken to also re-order inputs for better consistency.
  - `VecZnxNormalize` & `VecZnxNormalizeImpl`
  - `VecZnxBigNormalize` & `VecZnxBigNormalizeImpl`
  This change completes the road to unlocking full support for cross-base2k normalization, along with arbitrary positive/negative offset. Code is not ensured to be optimal, but correctness is ensured. 

## `poulpy-cpu-ref`
- Implemented `ConvolutionImpl` OPE on `FFT64Ref` backend.
- Add benchmark for convolution.
- Add test for convolution.

## `poulpy-cpu-avx`
- Implemented `ConvolutionImpl` OPE on `FFT64Avx` backend.
- Add benchmark for convolution.
- Add test for convolution.
- Add fft64 AVX code for
  - `reim4_save_1blk_to_reim_contiguous_avx`
  - `reim4_convolution_1coeff_avx`
  - `reim4_convolution_2coeffs_avx`

## `poulpy-core`
- Renamed `size` to `limbs`.
- Add `GLWEMulPlain` trait:
  - `glwe_mul_plain_tmp_bytes`
  - `glwe_mul_plain`
  - `glwe_mul_plain_inplace`
- Add `GLWEMulConst` trait:
  - `glwe_mul_const_tmp_bytes`
  - `glwe_mul_const`
  - `glwe_mul_const_inplace`
- Add `GLWETensoring` trait:
  - `glwe_tensor_apply_tmp_bytes`
  - `glwe_tensor_apply`
  - `glwe_tensor_relinearize_tmp_bytes`
  - `glwe_tensor_relinearize`
- Add method tests:
  - `test_glwe_tensoring`

## [0.4.0] - 2025-11-20

### Summary
- Full support for base2k operations.
- Many improvements to BDD arithmetic.
- Removal of **poulpy-backend** & spqlios backend.
- Addition of individual crates for each specific backend.
- Some minor bug fixes.

### `poulpy-hal`
- Add cross-base2k normalization

### `poulpy-core`
- Add full support for automatic cross-base2k operations & updated tests accordingly.
- Updated noise helper API.
- Fixed many tests that didn't assess noise correctly.
- Fixed decoding function to use arithmetic rounded division instead of arithmetic right shift.
- Fixed packing to clean values correctly.

### `poulpy-schemes`
- Renamed `tfhe` crate to `bin_fhe`.
- Improved support & API for BDD arithmetic, including multi-thread acceleration.
- Updated crate to support cross-base2k operations.
- Add additional operations, such as splice_u8, splice_u16 and sign extension.
- Add `GLWEBlindRetriever` and `GLWEBlindRetrieval`: a `GGSW`-based blind reversible retrieval (enables to instantiate encrypted ROM/RAM like object).
- Improved Cmux speed

### `poulpy-cpu-ref`
- A new crate that provides the reference CPU implementation of **poulpy-hal**. This replaces the previous **poulpy-backend/cpu_ref**.

### `poulpy-cpu-avx`
- A new crate that provides an AVX/FMA accelerated CPU implementation of **poulpy-hal**. This replaces the previous **poulpy-backend/cpu_avx**.

### `poulpy-schemes`
 - Added `sign` argument to GGSW-based blind rotation, which enables to choose the rotation direction of the test vector.

## [0.3.2] - 2025-10-27

### `poulpy-hal`
- Improved convolution functionality

### `poulpy-core`
 - Rename `GLWEToLWESwitchingKey` to `GLWEToLWEKey`.
 - Rename `LWEToGLWESwitchingKey` to `LWEToGLWEKey`.
 - Add `GLWESecretTensor` which stores the flattened upper right of the tensor matrix of the pairs  `sk[i] * sk[j]`.
 - Add `GGLWEToGGSWKey`, `GGLWEToGGSWKeyPrepared`, `GGLWEToGGSWKeyCompressed`, which encrypts the full tensor matrix of all pairs `sk[i] * sk[j]`, with one `GGLWE` per row.
 - Update `GGLWEToGGSW` API to take `GGLWEToGGSWKey` instead of the `GLWETensorKey`
 - Add `GLWETensor`, the result of tensoring two `GLWE` of identical rank.
 - Changed `GLWETensorKey` to be an encryption of `GLWESecretTensor` (preliminary work for `GLWEFromGLWETensor`, a.k.a relinearization). 

### `poulpy-schemes`
 - Add `GLWEBlindRotation`, a `GGSW`-based blind rotation that evaluates `GLWE <- GLWE * X^{((k>>bit_rsh) % 2^bit_mask) << bit_lsh}.` (`k` = `FheUintBlocksPrepared`).
 - Add `GGSWBlindRotation`, a `GGSW`-based blind rotation that evaluates `GGSW <- (GGSW or ScalarZnx) * X^{((k>>bit_rsh) % 2^bit_mask) << bit_lsh}.` (`k` = `FheUintBlocksPrepared`).

## [0.3.1] - 2025-10-24

### `poulpy-hal`
 - Add bivariate convolution (X, Y) / (X^{N} + 1) with Y = 2^-K

### `poulpy-core`
 - Fix typo in impl of GGLWEToRef for GLWEAutomorphismKey that required the data to be mutable.

## [0.3.0] - 2025-10-23

- Fixed builds on MACOS

### Breaking changes
 - The changes to `poulpy-core` required to break some of the existing API. For example the API `prepare_alloc` has been removed and the trait `Prepare<...>` has been broken down for each different ciphertext type (e.g. GLWEPrepare). To achieve the same functionality, the user must allocated the prepared ciphertext, and then call prepare on it.

### `poulpy-hal`
 - Added cross-base2k normalization

### `poulpy-core`
 - Added functionality-based traits, which removes the need to import the low-levels traits of `poulpy-hal` and makes backend agnostic code much cleaner. For example instead of having to import each individual traits required for the encryption of a GLWE, only the trait `GLWEEncryptSk` is needed.

### `poulpy-schemes`
 - Added basic framework for binary decision circuit (BDD) arithmetic along with some operations.

## [0.2.0] - 2025-09-15

### Breaking changes
 - Updated the trait `FillUniform` to take `log_bound`.

### `poulpy-hal`
 - Added pure Rust reference code for `vec_znx` and `fft64` backend.
 - Added cross-backend generic test suite along with macros.
 - Added benchmark generic test suite.

### `poulpy-backend`
 - Added `FFTRef` backend, which provides an implementation relying on the reference code of `poulpy-hal`.
 - Added `FFTAvx` backend, which provides a pure Rust AVX/FMA accelerated implementation of `FFTRef` backend.
 - Added cross-backend tests between `FFTRef` and `FFTAvx`.
 - Added cross-backend tests between `FFTRef` and `FFT64Spqlios`.

### `poulpy-core`
 - Removed unsafe blocks.
 - Added tests suite for `FFTRef` and `FFTAvx` backends.

### Other
 - Fixed a few minor bugs.

## [0.1.0] - 2025-08-25
 - Initial release.