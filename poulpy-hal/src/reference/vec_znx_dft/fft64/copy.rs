use rand::RngCore;

use crate::{
    api::VecZnxDftCopy,
    layouts::{Backend, DataViewMut, Module, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, ZnxInfos, ZnxView, ZnxViewMut},
    oep::VecZnxDftAllocBytesImpl,
    reference::reim::{ReimArithmetic, ReimArithmeticRef},
    source::Source,
};

pub fn vec_znx_dft_copy<R, A, BE, REIMARI>(step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = f64>,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
    REIMARI: ReimArithmetic,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), a.n())
    }

    let steps: usize = a.size().div_ceil(step);
    let min_steps: usize = res.size().min(steps);

    (0..min_steps).for_each(|j| {
        let limb: usize = offset + j * step;
        if limb < a.size() {
            REIMARI::reim_copy(res.at_mut(res_col, j), a.at(a_col, limb));
        }
    });
    (min_steps..res.size()).for_each(|j| {
        REIMARI::reim_zero(res.at_mut(res_col, j));
    })
}

pub fn test_vec_znx_copy<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftCopy<B>,
    B: Backend<ScalarPrep = f64> + VecZnxDftAllocBytesImpl<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, a_size);

        a.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_f64(-1., 1.));

        for res_size in [1, 2, 6, 11] {
            let mut res_0: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, res_size);
            let mut res_1: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, res_size);

            // Set d to garbage
            source.fill_bytes(res_0.data_mut());
            source.fill_bytes(res_1.data_mut());

            // Reference
            for i in 0..cols {
                vec_znx_dft_copy::<_, _, _, ReimArithmeticRef>(1, 0, &mut res_0, i, &a, i);
                module.vec_znx_dft_copy(1, 0, &mut res_1, i, &a, i);
            }

            assert_eq!(res_0.raw(), res_1.raw());

            // Set d to garbage
            source.fill_bytes(res_0.data_mut());
            source.fill_bytes(res_1.data_mut());

            // Reference
            for i in 0..cols {
                vec_znx_dft_copy::<_, _, _, ReimArithmeticRef>(1, 1, &mut res_0, i, &a, i);
                module.vec_znx_dft_copy(1, 1, &mut res_1, i, &a, i);
            }

            assert_eq!(res_0.raw(), res_1.raw());

            // Set d to garbage
            source.fill_bytes(res_0.data_mut());
            source.fill_bytes(res_1.data_mut());

            // Reference
            for i in 0..cols {
                vec_znx_dft_copy::<_, _, _, ReimArithmeticRef>(2, 1, &mut res_0, i, &a, i);
                module.vec_znx_dft_copy(2, 1, &mut res_1, i, &a, i);
            }

            assert_eq!(res_0.raw(), res_1.raw());
        }
    }
}
