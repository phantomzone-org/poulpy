use crate::{
    api::{VecZnxBigAdd, VecZnxBigAddInplace, VecZnxBigAddSmall, VecZnxBigAddSmallInplace},
    layouts::{
        Backend, FillUniform, Module, VecZnx, VecZnxBig, VecZnxBigToMut, VecZnxBigToRef, VecZnxToRef, ZnxView, ZnxViewMut,
    },
    oep::VecZnxBigAllocBytesImpl,
    reference::vec_znx::{vec_znx_add_inplace_ref, vec_znx_add_ref},
    source::Source,
};

pub fn vec_znx_big_add_ref<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend,
    R: VecZnxBigToMut<BE>,
    A: VecZnxBigToRef<BE>,
    B: VecZnxBigToRef<BE>,
{
    let res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnxBig<&[u8], BE> = a.to_ref();
    let b: VecZnxBig<&[u8], BE> = b.to_ref();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    let a_vznx: VecZnx<&[u8]> = VecZnx {
        data: a.data,
        n: a.n,
        cols: a.cols,
        size: a.size,
        max_size: a.max_size,
    };

    let b_vznx: VecZnx<&[u8]> = VecZnx {
        data: b.data,
        n: b.n,
        cols: b.cols,
        size: b.size,
        max_size: b.max_size,
    };

    vec_znx_add_ref(&mut res_vznx, res_col, &a_vznx, a_col, &b_vznx, b_col);
}

pub fn vec_znx_big_add_inplace_ref<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend,
    R: VecZnxBigToMut<BE>,
    A: VecZnxBigToRef<BE>,
{
    let res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnxBig<&[u8], BE> = a.to_ref();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    let a_vznx: VecZnx<&[u8]> = VecZnx {
        data: a.data,
        n: a.n,
        cols: a.cols,
        size: a.size,
        max_size: a.max_size,
    };

    vec_znx_add_inplace_ref(&mut res_vznx, res_col, &a_vznx, a_col);
}

pub fn vec_znx_big_add_small_ref<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend,
    R: VecZnxBigToMut<BE>,
    A: VecZnxBigToRef<BE>,
    B: VecZnxToRef,
{
    let res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnxBig<&[u8], BE> = a.to_ref();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    let a_vznx: VecZnx<&[u8]> = VecZnx {
        data: a.data,
        n: a.n,
        cols: a.cols,
        size: a.size,
        max_size: a.max_size,
    };

    vec_znx_add_ref(&mut res_vznx, res_col, &a_vznx, a_col, b, b_col);
}

pub fn vec_znx_big_add_small_inplace_ref<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend,
    R: VecZnxBigToMut<BE>,
    A: VecZnxToRef,
{
    let res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    vec_znx_add_inplace_ref(&mut res_vznx, res_col, a, a_col);
}

pub fn test_vec_znx_big_add<B: Backend<ScalarBig = i64>>(module: &Module<B>)
where
    Module<B>: VecZnxBigAdd<B>,
    B: VecZnxBigAllocBytesImpl<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, a_size);
        a.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        for b_size in [1, 2, 6, 11] {
            let mut b: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, b_size);
            b.raw_mut()
                .iter_mut()
                .for_each(|x| *x = source.next_i32() as i64);

            for res_size in [1, 2, 6, 11] {
                let mut res_0: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);
                let mut res_1: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);

                // Set d to garbage
                res_0.fill_uniform(&mut source);
                res_1.fill_uniform(&mut source);

                // Reference
                for i in 0..cols {
                    vec_znx_big_add_ref(&mut res_0, i, &a, i, &b, i);
                    module.vec_znx_big_add(&mut res_1, i, &a, i, &b, i);
                }

                assert_eq!(res_0.raw(), res_1.raw());
            }
        }
    }
}

pub fn test_vec_znx_big_add_inplace<B: Backend<ScalarBig = i64>>(module: &Module<B>)
where
    Module<B>: VecZnxBigAddInplace<B>,
    B: VecZnxBigAllocBytesImpl<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, a_size);
        a.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        for res_size in [1, 2, 6, 11] {
            let mut res_0: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);
            let mut res_1: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);

            res_0
                .raw_mut()
                .iter_mut()
                .for_each(|x| *x = source.next_i32() as i64);

            res_1.raw_mut().copy_from_slice(res_0.raw());

            for i in 0..cols {
                vec_znx_big_add_inplace_ref(&mut res_0, i, &a, i);
                module.vec_znx_big_add_inplace(&mut res_1, i, &a, i);
            }

            assert_eq!(res_0.raw(), res_1.raw());
        }
    }
}

pub fn test_vec_znx_big_add_small<B: Backend<ScalarBig = i64>>(module: &Module<B>)
where
    Module<B>: VecZnxBigAddSmall<B>,
    B: VecZnxBigAllocBytesImpl<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, a_size);
        a.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        for b_size in [1, 2, 6, 11] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, b_size);
            b.raw_mut()
                .iter_mut()
                .for_each(|x| *x = source.next_i32() as i64);

            for res_size in [1, 2, 6, 11] {
                let mut res_0: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);
                let mut res_1: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);

                // Set d to garbage
                res_0.fill_uniform(&mut source);
                res_1.fill_uniform(&mut source);

                // Reference
                for i in 0..cols {
                    vec_znx_big_add_small_ref(&mut res_0, i, &a, i, &b, i);
                    module.vec_znx_big_add_small(&mut res_1, i, &a, i, &b, i);
                }

                assert_eq!(res_0.raw(), res_1.raw());
            }
        }
    }
}

pub fn test_vec_znx_big_add_small_inplace<B: Backend<ScalarBig = i64>>(module: &Module<B>)
where
    Module<B>: VecZnxBigAddSmallInplace<B>,
    B: VecZnxBigAllocBytesImpl<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, a_size);
        a.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        for res_size in [1, 2, 6, 11] {
            let mut res_0: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);
            let mut res_1: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);

            res_0
                .raw_mut()
                .iter_mut()
                .for_each(|x| *x = source.next_i32() as i64);

            res_1.raw_mut().copy_from_slice(res_0.raw());

            for i in 0..cols {
                vec_znx_big_add_small_inplace_ref(&mut res_0, i, &a, i);
                module.vec_znx_big_add_small_inplace(&mut res_1, i, &a, i);
            }

            assert_eq!(res_0.raw(), res_1.raw());
        }
    }
}
