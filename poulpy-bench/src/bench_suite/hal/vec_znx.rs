//! Criterion benchmark harnesses for coefficient-domain [`VecZnx`] operations.
//!
//! Re-exports all `bench_vec_znx_*` functions from the individual
//! reference implementation modules so that backend crates can drive
//! them from a single import path.

pub use poulpy_cpu_ref::reference::vec_znx::{
    bench_vec_znx_add_assign, bench_vec_znx_add_into, bench_vec_znx_automorphism, bench_vec_znx_automorphism_assign,
    bench_vec_znx_lsh, bench_vec_znx_lsh_assign, bench_vec_znx_mul_xp_minus_one, bench_vec_znx_mul_xp_minus_one_assign,
    bench_vec_znx_negate, bench_vec_znx_negate_assign, bench_vec_znx_normalize, bench_vec_znx_normalize_assign,
    bench_vec_znx_rotate, bench_vec_znx_rotate_assign, bench_vec_znx_rsh, bench_vec_znx_rsh_assign, bench_vec_znx_sub,
    bench_vec_znx_sub_assign, bench_vec_znx_sub_negate_assign,
};
