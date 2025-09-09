use itertools::izip;

use crate::{
    api::VecZnxSwitchRing,
    layouts::{Backend, FillUniform, Module, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::{vec_znx::vec_znx_copy_ref, znx::znx_zero_ref},
    source::Source,
};

/// Maps between negacyclic rings by changing the polynomial degree.
/// Up:  Z[X]/(X^N+1) -> Z[X]/(X^{2^d N}+1) via X ↦ X^{2^d}
/// Down: Z[X]/(X^N+1) -> Z[X]/(X^{N/2^d}+1) by folding indices.
pub fn vec_znx_switch_ring_ref<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    let a: VecZnx<&[u8]> = a.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    let (n_in, n_out) = (a.n(), res.n());

    if n_in == n_out {
        vec_znx_copy_ref(&mut res, res_col, &a, a_col);
        return;
    }

    let (gap_in, gap_out): (usize, usize);
    if n_in > n_out {
        (gap_in, gap_out) = (n_in / n_out, 1)
    } else {
        (gap_in, gap_out) = (1, n_out / n_in);
        for j in 0..res.size() {
            znx_zero_ref(res.at_mut(res_col, j));
        }
    }

    let min_size: usize = a.size().min(res.size());

    (0..min_size).for_each(|i| {
        izip!(
            a.at(a_col, i).iter().step_by(gap_in),
            res.at_mut(res_col, i).iter_mut().step_by(gap_out)
        )
        .for_each(|(x_in, x_out)| *x_out = *x_in);
    });

    for j in min_size..res.size() {
        znx_zero_ref(res.at_mut(res_col, j));
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
                    vec_znx_switch_ring_ref(&mut r1, i, &a, i);
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
                    vec_znx_switch_ring_ref(&mut r1, i, &a, i);
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
