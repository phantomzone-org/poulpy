# 🐙 Poulpy-CPU-AVX

**Poulpy-Backend-CPU-AVX** is a Rust crate that provides the reference CPU implementation of **`poulpy-hal`**. This crate is used to instantiate projects implemented with **`poulpy-hal`**, **`poulpy-core`** and/or **`poulpy-schemes`**.

## Example

```rust
use poulpy_backend_cpu_ref::FFT64Ref;
use poulpy_hal::{api::ModuleNew, layouts::Module};

let log_n: usize = 10;
let module = Module<FFT64Ref> = Module<FFT64Ref>::new(1<<log_n);
```

## Contributors

To add your own backend, implement the open extension traits from **`poulpy-hal/oep`** for a struct that implements the `Backend` trait.  
This will automatically make your backend compatible with the API of **`poulpy-hal`**, **`poulpy-core`** and **`poulpy-schemes`**.