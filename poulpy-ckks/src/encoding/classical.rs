//! Classical CKKS encoding and decoding implementation.
//!
//! This module provides the implementation of standard polynomial
//! encoding via inverse canonical embedding.

use crate::layouts::plaintext::CKKSPlaintext;

/// Maximum log_delta allowed for encoding with an f64-based FFT backend.
///
/// FFT64 backends (both reference and AVX2) use IEEE-754 double precision
/// arithmetic internally. Their integer-to-float conversion routines require
/// polynomial coefficients to satisfy |x| < 2^50 (see `reim_from_znx_i64_bnd50`).
/// Choosing log_delta >= 50 would cause the scaled values delta * v to overflow
/// that bound, producing silent wrong results.
const F64_FFT_MAX_LOG_DELTA: u32 = 50;
use poulpy_hal::{
    api::{ModuleN, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VecZnxDftApply, VecZnxIdftApplyConsume},
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxInfos, ZnxView, ZnxViewMut},
};

/// Returns the minimum scratch-space size in bytes required by [`encode`].
pub fn encode_tmp_bytes<BE: Backend>(module: &Module<BE>) -> usize
where
    Module<BE>: VecZnxBigNormalizeTmpBytes,
{
    module.vec_znx_big_normalize_tmp_bytes()
}

/// Encodes complex slot values into a CKKS plaintext via inverse canonical embedding.
pub fn encode<BE: Backend<ScalarPrep = f64, ScalarBig = i64>>(
    module: &Module<BE>,
    pt: &mut CKKSPlaintext<impl DataMut>,
    re: &[f64],
    im: &[f64],
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: ModuleN + VecZnxDftAlloc<BE> + VecZnxIdftApplyConsume<BE> + VecZnxBigNormalize<BE> + VecZnxBigNormalizeTmpBytes,
{
    let n = module.n();
    let m = n / 2;

    assert!(
        pt.log_delta <= F64_FFT_MAX_LOG_DELTA,
        "log_delta ({}) exceeds the f64 FFT precision limit of {} bits",
        pt.log_delta,
        F64_FFT_MAX_LOG_DELTA
    );
    assert!(
        re.len() <= m,
        "Real part length {} exceeds the number of slots ({})",
        re.len(),
        m
    );
    assert!(
        im.len() <= m,
        "Imaginary part length {} exceeds number of slots ({})",
        im.len(),
        m
    );

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

    // Bivariate Normalization: Decompose the unreduced IDFT output into base-2^K limbs.
    module.vec_znx_big_normalize(&mut pt.inner.data, base2k, res_offset, 0, &big, base2k, 0, scratch);
}

/// Decodes a CKKS plaintext into complex slot values.
pub fn decode<BE: Backend<ScalarPrep = f64>>(module: &Module<BE>, pt: &CKKSPlaintext<impl DataRef>) -> (Vec<f64>, Vec<f64>)
where
    Module<BE>: ModuleN + VecZnxDftAlloc<BE> + VecZnxDftApply<BE>,
{
    assert!(
        pt.inner.k.0 >= pt.log_delta,
        "total precision k={} is less than log_delta ({}): \
         not enough limbs to represent the scaling factor",
        pt.inner.k.0,
        pt.log_delta
    );
    let n = module.n();
    let m = n / 2;
    let base2k: usize = pt.inner.base2k.into();
    let size = pt.inner.data.size();

    // Perform forward DFT on each limb independently.
    let mut dft = module.vec_znx_dft_alloc(1, size);
    module.vec_znx_dft_apply(1, 0, &mut dft, 0, &pt.inner.data, 0);

    let mut re = vec![0.0f64; m];
    let mut im = vec![0.0f64; m];

    // Horner evaluation of sum_l c_l / 2^{(l+1)*K}, processing from LSB to MSB.
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
