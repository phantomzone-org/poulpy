//! Criterion benchmark harnesses for coefficient-domain [`VecZnx`] operations.
//!
//! Re-exports all `bench_vec_znx_*` functions from the individual
//! reference implementation modules so that backend crates can drive
//! them from a single import path.

pub use crate::reference::vec_znx::{
    bench_vec_znx_add, bench_vec_znx_add_inplace, bench_vec_znx_automorphism, bench_vec_znx_automorphism_inplace,
    bench_vec_znx_lsh, bench_vec_znx_lsh_inplace, bench_vec_znx_mul_xp_minus_one, bench_vec_znx_mul_xp_minus_one_inplace,
    bench_vec_znx_negate, bench_vec_znx_negate_inplace, bench_vec_znx_normalize, bench_vec_znx_normalize_inplace,
    bench_vec_znx_rotate, bench_vec_znx_rotate_inplace, bench_vec_znx_rsh, bench_vec_znx_rsh_inplace, bench_vec_znx_sub,
    bench_vec_znx_sub_inplace, bench_vec_znx_sub_negate_inplace,
};
