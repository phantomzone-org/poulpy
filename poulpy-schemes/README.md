# 🐙 Poulpy-Schemes

**Poulpy-Schemes** is a Rust crate built on **`poulpy-hal`** and **`poulpy-core`**, providing backend agnostic RLWE-based FHE schemes implementation.

## Getting Started

See [./examples/circuit_bootstrapping.rs](./examples/circuit_bootstrapping.rs)

## Available Schemes

- **BIN FHE**:
  - **bdd_arithmetic**: high level API for u32 arithmetic (u8 to u256 planned) using binary decision circuits. Also provides API for blind retrieval, blind rotation (using encrypted integers) and blind selection.
  - **blind_rotation**: API for blind rotation (LWE(m) -> GLWE(X^m))
  - **circuit_bootstrapping**: API for circuit bootstrapping (LWE(m) -> GGSW(m) or GGSW(X^m)).
- **CKKS**: planned