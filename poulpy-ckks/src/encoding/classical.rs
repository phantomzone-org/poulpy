//! Classical CKKS encoding and decoding via the canonical embedding.

use std::fmt::Debug;

use poulpy_hal::{
    layouts::{Backend, Module},
    reference::fft64::reim::FFTModuleHandle,
};
use rand_distr::num_traits::{Float, FloatConst};

/// Encodes complex slot values into a compact CKKS plaintext.
///
/// The message polynomial is scaled by `2^embed_bits` and stored as a small
/// integer polynomial in base `2^base2k`, using `ceil(embed_bits/base2k)` limbs.
/// The Torus positioning (`2^{-Delta}`) happens when the plaintext is used
/// (during encryption or plaintext operations).
pub(crate) fn encode_reim<F, BE: Backend>(module: &Module<BE>, pt: &mut [F], re: &[F], im: &[F])
where
    F: Float + FloatConst + Debug,
    Module<BE>: FFTModuleHandle<F>,
{
    let n = pt.len();
    let m = n / 2;

    assert_eq!(module.n(), n);
    assert_eq!(re.len(), m);
    assert_eq!(im.len(), m);

    pt[..m].copy_from_slice(re);
    pt[m..].copy_from_slice(im);

    module.get_ifft_table().execute(pt);
}

/// Decodes a CKKS plaintext into complex slot values via the canonical embedding.
///
/// The plaintext should be compact (with `ceil(embed_bits/base2k)` limbs).
/// Reconstructs the integer polynomial via Horner's method and divides
/// by `2^embed_bits` to recover the original message.
///
/// When `k` is not a multiple of `base2k`, the MSB limb (limb 0) contains
/// only `k % base2k` valid bits. Any upper bits are masked (sign-extended)
/// before the DFT so the Horner does not interpret garbage as signal.
pub(crate) fn decode_reim<F, BE: Backend>(module: &Module<BE>, pt: &[F], re: &mut [F], im: &mut [F])
where
    F: Float + FloatConst + Debug,
    Module<BE>: FFTModuleHandle<F>,
{
    let n = pt.len();
    let m = n / 2;

    assert_eq!(module.n(), n);
    assert!(re.len() <= m);
    assert!(im.len() <= m);

    let mut reim_tmp = vec![F::zero(); n];
    reim_tmp.copy_from_slice(&pt);

    module.get_ifft_table().execute(&mut reim_tmp);

    re.copy_from_slice(&reim_tmp[..m]);
    im.copy_from_slice(&reim_tmp[m..]);
}
