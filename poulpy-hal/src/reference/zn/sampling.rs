use crate::{
    layouts::{Zn, ZnToMut, ZnxInfos, ZnxViewMut},
    reference::znx::{znx_add_normal_f64_ref, znx_fill_normal_f64_ref, znx_fill_uniform_ref},
    source::Source,
};

pub fn zn_fill_uniform<R>(n: usize, basek: usize, res: &mut R, res_col: usize, source: &mut Source)
where
    R: ZnToMut,
{
    let mut res: Zn<&mut [u8]> = res.to_mut();
    for j in 0..res.size() {
        znx_fill_uniform_ref(basek, &mut res.at_mut(res_col, j)[..n], source)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn zn_fill_normal<R>(
    n: usize,
    basek: usize,
    res: &mut R,
    res_col: usize,
    k: usize,
    source: &mut Source,
    sigma: f64,
    bound: f64,
) where
    R: ZnToMut,
{
    let mut res: Zn<&mut [u8]> = res.to_mut();
    assert!(
        (bound.log2().ceil() as i64) < 64,
        "invalid bound: ceil(log2(bound))={} > 63",
        (bound.log2().ceil() as i64)
    );

    let limb: usize = k.div_ceil(basek) - 1;
    let scale: f64 = (1 << ((limb + 1) * basek - k)) as f64;
    znx_fill_normal_f64_ref(
        &mut res.at_mut(res_col, limb)[..n],
        sigma * scale,
        bound * scale,
        source,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn zn_add_normal<R>(
    n: usize,
    basek: usize,
    res: &mut R,
    res_col: usize,
    k: usize,
    source: &mut Source,
    sigma: f64,
    bound: f64,
) where
    R: ZnToMut,
{
    let mut res: Zn<&mut [u8]> = res.to_mut();
    assert!(
        (bound.log2().ceil() as i64) < 64,
        "invalid bound: ceil(log2(bound))={} > 63",
        (bound.log2().ceil() as i64)
    );

    let limb: usize = k.div_ceil(basek) - 1;
    let scale: f64 = (1 << ((limb + 1) * basek - k)) as f64;
    znx_add_normal_f64_ref(
        &mut res.at_mut(res_col, limb)[..n],
        sigma * scale,
        bound * scale,
        source,
    )
}
