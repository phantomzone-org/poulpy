//! Classical CKKS encoding and decoding via the canonical embedding.

use crate::layouts::plaintext::CKKSPlaintext;

use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc,
        VecZnxDftApply, VecZnxIdftApplyConsume,
    },
    layouts::{DataMut, DataRef, Module, ScratchOwned, ZnxInfos, ZnxView, ZnxViewMut},
};

// The FFT64 AVX backend uses an IEEE 754 bit-manipulation trick for i64-to-f64
// conversion (reim_from_znx_i64_bnd50_fma) that requires |x| < 2^50. CKKS with
// NTT120 parameters (base2k=52) produces coefficients up to 52 bits, exceeding
// this bound. Force the ref backend until the AVX conversion supports wider inputs.
type Fft64Backend = poulpy_cpu_ref::FFT64Ref;

const F64_FFT_MAX_LOG_DELTA: u32 = 50;

/// Encodes complex slot values into a compact CKKS plaintext.
///
/// The message polynomial is scaled by `2^log_delta` and stored as a small
/// integer polynomial in base `2^base2k`, using `ceil(log_delta/base2k)` limbs.
/// The Torus positioning (`2^{-Delta}`) happens when the plaintext is used
/// (during encryption or plaintext operations).
pub fn encode(pt: &mut CKKSPlaintext<impl DataMut>, re: &[f64], im: &[f64]) {
    let n = pt.inner.data.n();
    let m = n / 2;

    assert!(pt.log_delta <= F64_FFT_MAX_LOG_DELTA);
    assert!(re.len() <= m);
    assert!(im.len() <= m);

    let module = Module::<Fft64Backend>::new(n as u64);

    let base2k: usize = pt.inner.base2k.into();
    let pt_size = pt.inner.data.size();
    // Place the integer at the bottom of the compact representation (no sub-limb shift).
    let res_offset: i64 = base2k as i64 - (pt_size * base2k) as i64;
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
///
/// The plaintext should be compact (with `ceil(log_delta/base2k)` limbs).
/// Reconstructs the integer polynomial via Horner's method and divides
/// by `2^log_delta` to recover the original message.
///
/// When `k` is not a multiple of `base2k`, the MSB limb (limb 0) contains
/// only `k % base2k` valid bits. Any upper bits are masked (sign-extended)
/// before the DFT so the Horner does not interpret garbage as signal.
pub fn decode(pt: &CKKSPlaintext<impl DataRef>) -> (Vec<f64>, Vec<f64>) {
    use poulpy_hal::layouts::VecZnx;

    assert!(pt.inner.k.0 >= pt.log_delta);

    let n = pt.inner.data.n();
    let m = n / 2;
    let base2k: usize = pt.inner.base2k.into();
    let k: usize = pt.inner.k.0 as usize;
    let size = pt.inner.data.size();
    let top_limb_bits = if k.is_multiple_of(base2k) { base2k } else { k % base2k };

    let module = Module::<Fft64Backend>::new(n as u64);

    // When k is not a multiple of base2k, the MSB limb may have garbage
    // above the valid bits (e.g. after glwe_rsh). Mask before DFT.
    let needs_masking = top_limb_bits < base2k;
    let masked = if needs_masking {
        let mut m = VecZnx::alloc(n, 1, size);
        for l in 0..size {
            let src: &[i64] = pt.inner.data.at(0, l);
            let dst: &mut [i64] = m.at_mut(0, l);
            dst.copy_from_slice(src);
        }
        let mask_shift = 64 - top_limb_bits;
        let limb0: &mut [i64] = m.at_mut(0, 0);
        for v in limb0.iter_mut() {
            *v = (*v << mask_shift) >> mask_shift;
        }
        Some(m)
    } else {
        None
    };

    let mut dft = module.vec_znx_dft_alloc(1, size);
    if let Some(ref m) = masked {
        module.vec_znx_dft_apply(1, 0, &mut dft, 0, m, 0);
    } else {
        module.vec_znx_dft_apply(1, 0, &mut dft, 0, &pt.inner.data, 0);
    }

    let mut re = vec![0.0f64; m];
    let mut im = vec![0.0f64; m];

    // Reconstruct the integer polynomial using Horner's method from MSB (limb 0)
    // to LSB (limb size-1). Uniform 2^{base2k} scale for all limbs — the MSB
    // masking above ensures the top limb has only `top_limb_bits` valid bits,
    // so the Horner correctly produces H < 2^k.
    let scale_up = (2.0f64).powi(base2k as i32);
    for l in 0..size {
        let reim: &[f64] = dft.at(0, l);
        for j in 0..m {
            re[j] = re[j] * scale_up + reim[j];
            im[j] = im[j] * scale_up + reim[m + j];
        }
    }

    let inv_delta = (2.0f64).powi(-(pt.log_delta as i32));
    for j in 0..m {
        re[j] *= inv_delta;
        im[j] *= inv_delta;
    }

    (re, im)
}
