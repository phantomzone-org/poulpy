use crate::{
    api::{VecZnxAddScalar, VecZnxAddScalarInplace},
    layouts::{
        Backend, FillUniform, Module, ScalarZnx, ScalarZnxToRef, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut,
    },
    reference::znx::{znx_add_i64_ref, znx_add_inplace_i64_ref, znx_copy_ref, znx_zero_ref},
    source::Source,
};

pub fn vec_znx_add_scalar_ref<R, A, B>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize, b_limb: usize)
where
    R: VecZnxToMut,
    A: ScalarZnxToRef,
    B: VecZnxToRef,
{
    let a: ScalarZnx<&[u8]> = a.to_ref();
    let b: VecZnx<&[u8]> = b.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    let min_size: usize = b.size().min(res.size());

    #[cfg(debug_assertions)]
    {
        assert!(
            b_limb < min_size,
            "b_limb: {} > min_size: {}",
            b_limb,
            min_size
        );
    }

    for j in 0..min_size {
        if j == b_limb {
            znx_add_i64_ref(res.at_mut(res_col, j), a.at(a_col, 0), b.at(b_col, j));
        } else {
            znx_copy_ref(res.at_mut(res_col, j), b.at(b_col, j));
        }
    }

    for j in min_size..res.size() {
        znx_zero_ref(res.at_mut(res_col, j));
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn vec_znx_add_scalar_avx<R, A, B>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize, b_limb: usize)
where
    R: VecZnxToMut,
    A: ScalarZnxToRef,
    B: VecZnxToRef,
{
    use crate::reference::znx::znx_add_i64_avx;
    use crate::reference::znx::znx_copy_ref;

    let a: ScalarZnx<&[u8]> = a.to_ref();
    let b: VecZnx<&[u8]> = b.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    let min_size: usize = a.size().min(res.size());

    #[cfg(debug_assertions)]
    {
        assert!(b_limb < min_size);
    }

    for j in 0..min_size {
        if j == b_limb {
            znx_add_i64_avx(res.at_mut(res_col, j), a.at(a_col, 0), b.at(b_col, j));
        } else {
            znx_copy_ref(res.at_mut(res_col, j), b.at(b_col, j));
        }
    }

    for j in min_size..res.size() {
        znx_zero_ref(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_add_scalar_inplace_ref<R, A>(res: &mut R, res_col: usize, res_limb: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: ScalarZnxToRef,
{
    let a: ScalarZnx<&[u8]> = a.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert!(res_limb < res.size());
    }

    znx_add_inplace_i64_ref(res.at_mut(res_col, res_limb), a.at(a_col, 0));
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn vec_znx_add_scalar_inplace_avx<R, A>(res: &mut R, res_col: usize, res_limb: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: ScalarZnxToRef,
{
    use crate::reference::znx::znx_add_inplace_i64_avx;

    let a: ScalarZnx<&[u8]> = a.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert!(res_limb < res.size());
    }

    znx_add_inplace_i64_avx(res.at_mut(res_col, res_limb), a.at(a_col, 0));
}

pub fn test_vec_znx_add_scalar<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxAddScalar,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut a: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(module.n(), cols);
    a.raw_mut()
        .iter_mut()
        .for_each(|x| *x = source.next_i32() as i64);

    for a_size in [1, 2, 6, 11] {
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, a_size);
        b.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        for res_size in [1, 2, 6, 11] {
            let mut res_0: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);
            let mut res_1: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);

            // Set d to garbage
            res_0.fill_uniform(&mut source);
            res_1.fill_uniform(&mut source);

            // Reference
            for i in 0..cols {
                vec_znx_add_scalar_ref(&mut res_0, i, &a, i, &b, i, (res_size.min(a_size)) - 1);
                module.vec_znx_add_scalar(&mut res_1, i, &a, i, &b, i, (res_size.min(a_size)) - 1);
            }

            assert_eq!(res_0.raw(), res_1.raw());
        }
    }
}

pub fn test_vec_znx_add_scalar_inplace<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxAddScalarInplace,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for res_size in [1, 2, 6, 11] {
        let mut res_0: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);
        let mut res_1: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);

        let mut b: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(module.n(), cols);
        b.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        res_0
            .raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        res_1.raw_mut().copy_from_slice(res_0.raw());

        for i in 0..cols {
            vec_znx_add_scalar_inplace_ref(&mut res_0, i, res_size - 1, &b, i);
            module.vec_znx_add_scalar_inplace(&mut res_1, i, res_size - 1, &b, i);
        }

        assert_eq!(res_0.raw(), res_1.raw());
    }
}
