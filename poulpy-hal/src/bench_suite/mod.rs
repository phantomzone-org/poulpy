//! Criterion-based benchmark harnesses, generic over any backend.
//!
//! Backend crates call these functions to measure performance of their
//! implementations against various ring degrees and parameter sets.

pub mod convolution;
pub mod svp;
pub mod vec_znx;
pub mod vec_znx_big;
pub mod vec_znx_dft;
pub mod vmp;
