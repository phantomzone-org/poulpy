use crate::{
    api::VecZnxCopy,
    layouts::{Backend, FillUniform, Module, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::znx::{znx_copy_ref, znx_zero_ref},
    source::Source,
};

pub fn vec_znx_copy_ref<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), a.n())
    }

    let res_size = res.size();
    let a_size = a.size();

    let min_size = res_size.min(a_size);

    for j in 0..min_size {
        znx_copy_ref(res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res_size {
        znx_zero_ref(res.at_mut(res_col, j));
    }
}

pub fn test_vec_znx_copy<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxCopy,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, a_size);
        a.fill_uniform(&mut source);

        for res_size in [1, 2, 6, 11] {
            let mut res_0: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);
            let mut res_1: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);

            // Set d to garbage
            res_0.fill_uniform(&mut source);
            res_1.fill_uniform(&mut source);

            // Reference
            for i in 0..cols {
                vec_znx_copy_ref(&mut res_0, i, &a, i);
                module.vec_znx_copy(&mut res_1, i, &a, i);
            }

            assert_eq!(res_0.raw(), res_1.raw());
        }
    }
}
