# Changelog

All notable changes to this project will be documented in this file.
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-08-21

- Initial release of the `vec_znx` (except convolution products), `vec_rnx` and `zn` apis.
- Hardware acceleration available: AVX2 (most parts)
- APIs are documented in the wiki and are in "beta mode": during the 2.x -> 3.x transition, functions whose API is satisfactory in test projects will pass in "stable mode".

## [1.0.0] - 2023-07-18

- Initial release of the double precision fft on the reim and cplx backends
- Coeffs-space conversions cplx <-> znx32 and tnx32
- FFT-space conversions cplx <-> reim4 layouts
- FFT-space multiplications on the cplx, reim and reim4 layouts.
- In this first release, the only platform supported is linux x86_64 (generic C code, and avx2/fma). It compiles on arm64, but without any acceleration.
