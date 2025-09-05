use std::f64::consts::SQRT_2;

use crate::{
    api::VecZnxBigAddNormal,
    layouts::{Backend, Module, VecZnxBig, VecZnxBigToMut, ZnxView, ZnxViewMut},
    oep::VecZnxBigAllocBytesImpl,
    reference::znx::znx_add_normal_f64_ref,
    source::Source,
};

pub fn vec_znx_big_add_normal_ref<R, B: Backend + Backend<ScalarBig = i64>>(
    basek: usize,
    res: &mut R,
    res_col: usize,
    k: usize,
    sigma: f64,
    bound: f64,
    source: &mut Source,
) where
    R: VecZnxBigToMut<B>,
{
    let mut res: VecZnxBig<&mut [u8], B> = res.to_mut();
    assert!(
        (bound.log2().ceil() as i64) < 64,
        "invalid bound: ceil(log2(bound))={} > 63",
        (bound.log2().ceil() as i64)
    );

    let limb: usize = k.div_ceil(basek) - 1;
    let scale: f64 = (1 << ((limb + 1) * basek - k)) as f64;
    znx_add_normal_f64_ref(
        res.at_mut(res_col, limb),
        sigma * scale,
        bound * scale,
        source,
    )
}

pub fn test_vec_znx_big_add_normal<B>(module: &Module<B>)
where
    Module<B>: VecZnxBigAddNormal<B>,
    B: Backend + Backend<ScalarBig = i64> + VecZnxBigAllocBytesImpl<B>,
{
    let n: usize = module.n();
    let basek: usize = 17;
    let k: usize = 2 * 17;
    let size: usize = 5;
    let sigma: f64 = 3.2;
    let bound: f64 = 6.0 * sigma;
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let zero: Vec<i64> = vec![0; n];
    let k_f64: f64 = (1u64 << k as u64) as f64;
    let sqrt2: f64 = SQRT_2;
    (0..cols).for_each(|col_i| {
        let mut a: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(n, cols, size);
        module.vec_znx_big_add_normal(basek, &mut a, col_i, k, &mut source, sigma, bound);
        module.vec_znx_big_add_normal(basek, &mut a, col_i, k, &mut source, sigma, bound);
        (0..cols).for_each(|col_j| {
            if col_j != col_i {
                (0..size).for_each(|limb_i| {
                    assert_eq!(a.at(col_j, limb_i), zero);
                })
            } else {
                let std: f64 = a.std(basek, col_i) * k_f64;
                assert!(
                    (std - sigma * sqrt2).abs() < 0.1,
                    "std={} ~!= {}",
                    std,
                    sigma * sqrt2
                );
            }
        })
    });
}
