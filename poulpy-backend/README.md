# 🐙 Poulpy-Backend

**Poulpy-Backend** is a Rust crate that provides concrete implementations of **`poulpy-hal`**. This crate is used to instantiate projects implemented with **`poulpy-hal`**, **`poulpy-core`** and/or **`poulpy-schemes`**.

## Backends

### cpu-spqlios

This module provides a CPU AVX2 accelerated backend through C bindings over [**spqlios-arithmetic**](https://github.com/tfhe/spqlios-arithmetic).

- Currently supported: `FFT64` backend  
- Planned: `NTT120` backend

### Build Notes

This backend is built and compiled automatically and has been tested on wsl/ubuntu.

- `cmake` is invoked automatically by the build script (`build.rs`) when compiling the crate.  
- No manual setup is required beyond having a standard Rust toolchain. 
- Build options can be changed in `/build/cpu_spqlios.rs`
- Automatic build of cpu-spqlios/spqlios-arithmetic can be disabled in `build.rs`.

Spqlios-arithmetic is windows/mac compatible but building for those platforms is slightly different (see [spqlios-arithmetic/wiki/build](https://github.com/tfhe/spqlios-arithmetic/wiki/build)) and has not been tested in Poulpy.

### Example

```rust
use poulpy_backend::cpu_spqlios::FFT64;
use poulpy_hal::{api::ModuleNew, layouts::Module};

let log_n: usize = 10;
let module = Module<FFT64> = Module<FFT64>::new(1<<log_n);
```

## Contributors

To add a backend, implement the open extension traits from **`poulpy-hal/oep`** for a struct that implements the `Backend` trait.  
This will automatically make your backend compatible with the API of **`poulpy-hal`**, **`poulpy-core`** and **`poulpy-schemes`**.