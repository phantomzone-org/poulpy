# CHANGELOG

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