use crate::{
    layouts::{NoiseInfos, VecZnx, VecZnxToMut, ZnxInfos, ZnxViewMut},
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

pub fn vec_znx_fill_normal_ref<R>(base2k: usize, res: &mut R, res_col: usize, noise_infos: NoiseInfos, source: &mut Source)
where
    R: VecZnxToMut,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    assert!(
        (noise_infos.bound.log2().ceil() as i64) < 64,
        "invalid bound: ceil(log2(bound))={} > 63",
        (noise_infos.bound.log2().ceil() as i64)
    );

    let (limb, scale) = noise_infos.target_limb_and_scale(base2k);
    znx_fill_normal_f64_ref(
        res.at_mut(res_col, limb),
        noise_infos.sigma * scale,
        noise_infos.bound * scale,
        source,
    )
}

pub fn vec_znx_add_normal_ref<R>(base2k: usize, res: &mut R, res_col: usize, noise_infos: NoiseInfos, source: &mut Source)
where
    R: VecZnxToMut,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    assert!(
        (noise_infos.bound.log2().ceil() as i64) < 64,
        "invalid bound: ceil(log2(bound))={} > 63",
        (noise_infos.bound.log2().ceil() as i64)
    );

    let (limb, scale) = noise_infos.target_limb_and_scale(base2k);
    znx_add_normal_f64_ref(
        res.at_mut(res_col, limb),
        noise_infos.sigma * scale,
        noise_infos.bound * scale,
        source,
    )
}
