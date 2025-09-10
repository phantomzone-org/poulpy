use crate::{
    api::VecZnxSwitchRing,
    layouts::{Backend, FillUniform, Module, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::{
        vec_znx::vec_znx_copy,
        znx::{ZnxArithmetic, ZnxArithmeticRef},
    },
    source::Source,
};

/// Maps between negacyclic rings by changing the polynomial degree.
/// Up:  Z[X]/(X^N+1) -> Z[X]/(X^{2^d N}+1) via X ↦ X^{2^d}
/// Down: Z[X]/(X^N+1) -> Z[X]/(X^{N/2^d}+1) by folding indices.
pub fn vec_znx_switch_ring<R, A, ZNXARI>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxArithmetic,
{
    let a: VecZnx<&[u8]> = a.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    let (n_in, n_out) = (a.n(), res.n());

    if n_in == n_out {
        vec_znx_copy::<_, _, ZNXARI>(&mut res, res_col, &a, a_col);
        return;
    }

    let min_size: usize = a.size().min(res.size());

    for j in 0..min_size {
        ZNXARI::znx_switch_ring(res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res.size() {
        ZNXARI::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn test_vec_znx_switch_ring<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxSwitchRing<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, a_size);

        // Fill a with random i64
        a.fill_uniform(&mut source);

        for res_size in [1, 2, 6, 11] {
            {
                let mut r0: VecZnx<Vec<u8>> = VecZnx::alloc(module.n() << 1, cols, res_size);
                let mut r1: VecZnx<Vec<u8>> = VecZnx::alloc(module.n() << 1, cols, res_size);

                r0.fill_uniform(&mut source);
                r1.fill_uniform(&mut source);

                // Normalize on c
                for i in 0..cols {
                    module.vec_znx_switch_ring(&mut r0, i, &a, i);
                    vec_znx_switch_ring::<_, _, ZnxArithmeticRef>(&mut r1, i, &a, i);
                }

                for i in 0..cols {
                    for j in 0..res_size {
                        assert_eq!(r0.at(i, j), r1.at(i, j));
                    }
                }
            }

            {
                let mut r0: VecZnx<Vec<u8>> = VecZnx::alloc(module.n() >> 1, cols, res_size);
                let mut r1: VecZnx<Vec<u8>> = VecZnx::alloc(module.n() >> 1, cols, res_size);

                r0.fill_uniform(&mut source);
                r1.fill_uniform(&mut source);

                // Normalize on c
                for i in 0..cols {
                    module.vec_znx_switch_ring(&mut r0, i, &a, i);
                    vec_znx_switch_ring::<_, _, ZnxArithmeticRef>(&mut r1, i, &a, i);
                }

                for i in 0..cols {
                    for j in 0..res_size {
                        assert_eq!(r0.at(i, j), r1.at(i, j));
                    }
                }
            }
        }
    }
}
