//! Classical CKKS encoding and decoding via the canonical embedding.

use crate::layouts::plaintext::CKKSPlaintext;

use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc,
        VecZnxDftApply, VecZnxIdftApplyConsume,
    },
    layouts::{DataMut, DataRef, Module, ScratchOwned, ZnxInfos, ZnxView, ZnxViewMut},
};

#[cfg(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
type Fft64Backend = poulpy_cpu_avx::FFT64Avx;
#[cfg(not(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
)))]
type Fft64Backend = poulpy_cpu_ref::FFT64Ref;

const F64_FFT_MAX_LOG_DELTA: u32 = 50;

/// Encodes complex slot values into a CKKS plaintext via inverse canonical embedding.
pub fn encode(pt: &mut CKKSPlaintext<impl DataMut>, re: &[f64], im: &[f64]) {
    let n = pt.inner.data.n();
    let m = n / 2;

    assert!(pt.log_delta <= F64_FFT_MAX_LOG_DELTA);
    assert!(re.len() <= m);
    assert!(im.len() <= m);

    let module = Module::<Fft64Backend>::new(n as u64);

    let base2k: usize = pt.inner.base2k.into();
    let res_offset: i64 = (base2k as i64) - (pt.log_delta as i64);
    let delta = (2.0f64).powi(pt.log_delta as i32);

    let mut dft = module.vec_znx_dft_alloc(1, 1);
    {
        let reim: &mut [f64] = dft.at_mut(0, 0);
        for j in 0..re.len() {
            reim[j] = delta * re[j];
        }
        for j in 0..im.len() {
            reim[m + j] = delta * im[j];
        }
    }
    let big = module.vec_znx_idft_apply_consume(dft);

    let mut scratch = ScratchOwned::<Fft64Backend>::alloc(module.vec_znx_big_normalize_tmp_bytes());
    module.vec_znx_big_normalize(&mut pt.inner.data, base2k, res_offset, 0, &big, base2k, 0, scratch.borrow());
}

/// Decodes a CKKS plaintext into complex slot values via the canonical embedding.
pub fn decode(pt: &CKKSPlaintext<impl DataRef>) -> (Vec<f64>, Vec<f64>) {
    assert!(pt.inner.k.0 >= pt.log_delta);

    let n = pt.inner.data.n();
    let m = n / 2;
    let base2k: usize = pt.inner.base2k.into();
    let size = pt.inner.data.size();

    let module = Module::<Fft64Backend>::new(n as u64);

    let mut dft = module.vec_znx_dft_alloc(1, size);
    module.vec_znx_dft_apply(1, 0, &mut dft, 0, &pt.inner.data, 0);

    let mut re = vec![0.0f64; m];
    let mut im = vec![0.0f64; m];

    let inv_scale = (2.0f64).powi(-(base2k as i32));
    for l in (0..size).rev() {
        let reim: &[f64] = dft.at(0, l);
        for j in 0..m {
            re[j] = (re[j] + reim[j]) * inv_scale;
            im[j] = (im[j] + reim[m + j]) * inv_scale;
        }
    }
    (re, im)
}
