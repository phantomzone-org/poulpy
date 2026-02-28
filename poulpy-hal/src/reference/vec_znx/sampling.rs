use crate::{
    layouts::{VecZnx, VecZnxToMut, ZnxInfos, ZnxViewMut},
    reference::znx::{znx_add_normal_f64_ref, znx_fill_normal_f64_ref, znx_fill_uniform_ref},
    source::Source,
};

pub fn vec_znx_fill_uniform_ref<R>(base2k: usize, res: &mut R, res_col: usize, source: &mut Source)
where
    R: VecZnxToMut,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    for j in 0..res.size() {
        znx_fill_uniform_ref(base2k, res.at_mut(res_col, j), source)
    }
}

pub fn vec_znx_fill_normal_ref<R>(
    base2k: usize,
    res: &mut R,
    res_col: usize,
    k: usize,
    sigma: f64,
    bound: f64,
    source: &mut Source,
) where
    R: VecZnxToMut,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    assert!(
        (bound.log2().ceil() as i64) < 64,
        "invalid bound: ceil(log2(bound))={} > 63",
        (bound.log2().ceil() as i64)
    );

    let limb: usize = k.div_ceil(base2k) - 1;
    let scale: f64 = (1u64 << ((limb + 1) * base2k - k)) as f64;
    znx_fill_normal_f64_ref(res.at_mut(res_col, limb), sigma * scale, bound * scale, source)
}

pub fn vec_znx_add_normal_ref<R>(
    base2k: usize,
    res: &mut R,
    res_col: usize,
    k: usize,
    sigma: f64,
    bound: f64,
    source: &mut Source,
) where
    R: VecZnxToMut,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    assert!(
        (bound.log2().ceil() as i64) < 64,
        "invalid bound: ceil(log2(bound))={} > 63",
        (bound.log2().ceil() as i64)
    );

    let limb: usize = k.div_ceil(base2k) - 1;
    let scale: f64 = (1u64 << ((limb + 1) * base2k - k)) as f64;
    znx_add_normal_f64_ref(res.at_mut(res_col, limb), sigma * scale, bound * scale, source)
}
