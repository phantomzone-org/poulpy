//! CKKS ciphertext multiplication.
//!
//! Multiplication in the bivariate representation is a convolution of the
//! base-`2^{base2k}` digit arrays.  The implementation exposes this as a
//! three-step pipeline: tensor → relinearize → rescale.
//!
//! The tensor offset determines which high-degree part of the bivariate
//! product is retained so that the result lands in the correct active torus
//! window:
//!
//! - ciphertext × ciphertext: `offset = max(k_a, k_b)`
//! - ciphertext × plaintext:  `offset = k_ct`
//!
//! After the product, the result metadata (`k`, `size`, sign extension) is
//! adjusted by [`set_active_k_and_size`], and [`rescale`] consumes
//! `min(log_delta_a, log_delta_b)` bits of precision.
//!
//! ## Variants
//!
//! | Function | Operands | Rescale |
//! |----------|----------|---------|
//! | [`mul`] | ct × ct | yes |
//! | [`mul_pt`] / [`mul_pt_inplace`] | ct × compact pt | yes |
//! | [`mul_prepared_pt`] / [`mul_prepared_pt_inplace`] | ct × prepared pt | yes |
//! | [`mul_const`] / [`mul_const_inplace`] | ct × complex constant | yes |
//! | [`mul_int`] / [`mul_int_inplace`] | ct × integer | no |

use crate::{
    layouts::{
        ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext, plaintext_prepared::CKKSPlaintextPrepared, tensor::CKKSTensor,
    },
    leveled::operations::{
        level::rescale,
        utils::{const_pt_from_scratch, const_pt_scratch_bytes, offset_pt_from_scratch, sign_extend_msb},
    },
};
use poulpy_core::{
    GLWEMulConst, GLWEMulPlain, GLWETensoring, ScratchTakeCore,
    layouts::{GLWEInfos, GLWELayout, GLWETensor, GLWETensorKeyPrepared, LWEInfos, Rank, SetGLWEInfos, TorusPrecision},
};
use poulpy_hal::{
    api::{VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

/// Sets the active precision window after multiplication or relinearization.
///
/// Updates `k`, shrinks `data.size` to the active limb count, and
/// sign-extends the MSB limb if `k` is not limb-aligned.
fn set_active_k_and_size(res: &mut CKKSCiphertext<impl DataMut>, k_eff: u32, size: usize) {
    debug_assert!(size <= res.inner.data_mut().max_size);
    res.inner.set_k(TorusPrecision(k_eff));
    res.inner.data_mut().size = size;
    sign_extend_msb(res);
}

/// Returns the scratch bytes needed for [`tensor`].
pub fn tensor_tmp_bytes<BE: Backend>(
    module: &Module<BE>,
    res: &CKKSTensor<impl Data>,
    a: &CKKSCiphertext<impl Data>,
    b: &CKKSCiphertext<impl Data>,
) -> usize
where
    Module<BE>: GLWETensoring<BE>,
{
    let off = a.inner.max_k().as_usize().max(b.inner.max_k().as_usize());
    module.glwe_tensor_apply_tmp_bytes(&res.inner, off, &a.inner, &b.inner)
}

/// Computes the tensor product of two CKKS ciphertexts.
///
/// The bivariate convolution offset is `max(k_a, k_b)`.  The result
/// `log_delta` is the sum of the operands' scaling factors.
pub fn tensor<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSTensor<impl DataMut>,
    a: &CKKSCiphertext<impl DataRef>,
    b: &CKKSCiphertext<impl DataRef>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWETensoring<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    assert_eq!(a.inner.base2k(), b.inner.base2k(), "tensor: base2k mismatch");
    let off = a.inner.max_k().as_usize().max(b.inner.max_k().as_usize());
    res.log_delta = a.log_delta + b.log_delta;
    module.glwe_tensor_apply(&mut res.inner, off, &a.inner, &b.inner, scratch);
}

/// Returns the scratch bytes needed for [`relinearize`].
pub fn relinearize_tmp_bytes<BE: Backend>(
    module: &Module<BE>,
    res: &CKKSCiphertext<impl Data>,
    tensor: &CKKSTensor<impl Data>,
    tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
) -> usize
where
    Module<BE>: GLWETensoring<BE>,
{
    module.glwe_tensor_relinearize_tmp_bytes(&res.inner, &tensor.inner, tsk)
}

/// Relinearizes a tensor product back into a rank-1 CKKS ciphertext.
///
/// After relinearization, sets the active precision window to `k_eff` / `size`
/// and sign-extends the MSB limb.
pub fn relinearize<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    tensor: &CKKSTensor<impl DataRef>,
    k_eff: u32,
    size: usize,
    tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWETensoring<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    module.glwe_tensor_relinearize(&mut res.inner, &tensor.inner, tsk, tsk.size(), scratch);
    res.log_delta = tensor.log_delta;
    set_active_k_and_size(res, k_eff, size);
}

/// Returns the scratch bytes needed for [`mul`].
pub fn mul_tmp_bytes<BE: Backend>(
    module: &Module<BE>,
    a: &CKKSCiphertext<impl Data>,
    b: &CKKSCiphertext<impl Data>,
    tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
) -> usize
where
    Module<BE>: GLWETensoring<BE>,
{
    let off = a.inner.max_k().as_usize().max(b.inner.max_k().as_usize());
    let mut layout = a.inner.glwe_layout();
    layout.k = TorusPrecision(a.inner.base2k().0 * a.inner.size() as u32);
    let tensor_bytes = GLWETensor::bytes_of(a.inner.n(), a.inner.base2k(), layout.k, Rank(1));
    let op_bytes = module
        .glwe_tensor_apply_tmp_bytes(&layout, off, &a.inner, &b.inner)
        .max(module.glwe_tensor_relinearize_tmp_bytes(&layout, &layout, tsk));
    tensor_bytes + op_bytes
}

/// Multiplies two CKKS ciphertexts: `res = a * b`.
///
/// Runs the full tensor → relinearize → rescale pipeline.  The rescale
/// consumes `min(log_delta_a, log_delta_b)` bits.  The result active
/// precision is `min(k_a, k_b) - rescale_bits`.
pub fn mul<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    a: &CKKSCiphertext<impl DataRef>,
    b: &CKKSCiphertext<impl DataRef>,
    tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWETensoring<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let rescale_bits = a.log_delta.min(b.log_delta);
    let k_eff = a.inner.max_k().0.min(b.inner.max_k().0);
    let size = a.inner.size().min(b.inner.size());
    let tensor_layout = GLWELayout {
        n: a.inner.n(),
        base2k: a.inner.base2k(),
        k: TorusPrecision(a.inner.base2k().0 * size as u32),
        rank: Rank(1),
    };
    let (tensor_inner, scratch_rest) = scratch.take_glwe_tensor(&tensor_layout);
    let mut tensor_res = CKKSTensor {
        inner: tensor_inner,
        log_delta: 0,
    };
    tensor(module, &mut tensor_res, a, b, scratch_rest);
    relinearize(module, res, &tensor_res, k_eff, size, tsk, scratch_rest);
    rescale(module, res, rescale_bits, scratch_rest);
}

/// Returns the scratch bytes needed for [`mul_pt`] and [`mul_pt_inplace`].
pub fn mul_pt_tmp_bytes<BE: Backend>(module: &Module<BE>, ct: &CKKSCiphertext<impl Data>) -> usize
where
    Module<BE>: GLWEMulPlain<BE> + VecZnxNormalizeTmpBytes,
{
    let mut layout = ct.inner.glwe_layout();
    layout.k = TorusPrecision(ct.inner.base2k().0 * ct.inner.size() as u32);
    let off = ct.inner.max_k().as_usize();
    let layout_bytes = poulpy_core::layouts::GLWEPlaintext::bytes_of(ct.inner.n(), ct.inner.base2k(), layout.k);
    let op_bytes = module.glwe_mul_plain_tmp_bytes(&layout, off, &ct.inner, &layout);
    layout_bytes + module.vec_znx_normalize_tmp_bytes().max(op_bytes)
}

/// Multiplies a ciphertext by a compact plaintext: `res = ct * pt`.
///
/// The compact plaintext is first expanded into the ciphertext torus layout.
/// Offset is `k_ct`.  Rescale consumes `min(log_delta_ct, log_delta_pt)` bits.
pub fn mul_pt<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    ct: &CKKSCiphertext<impl DataRef>,
    pt: &CKKSPlaintext<impl DataRef>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEMulPlain<BE> + VecZnxNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let size = ct.inner.size();
    let mk = TorusPrecision(ct.inner.base2k().0 * ct.inner.size() as u32);
    let off = ct.inner.max_k().as_usize();
    let (full_pt, scratch_rest) =
        offset_pt_from_scratch(module, ct.inner.n(), ct.inner.base2k(), mk, ct.inner.max_k(), pt, scratch);
    module.glwe_mul_plain(&mut res.inner, off, &ct.inner, &full_pt, scratch_rest);
    res.log_delta = ct.log_delta + pt.log_delta;
    set_active_k_and_size(res, ct.inner.max_k().0, size);
    let rescale_bits = ct.log_delta.min(pt.log_delta);
    rescale(module, res, rescale_bits, scratch_rest);
}

/// Multiplies a ciphertext by a compact plaintext in place: `ct *= pt`.
pub fn mul_pt_inplace<BE: Backend>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertext<impl DataMut>,
    pt: &CKKSPlaintext<impl DataRef>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEMulPlain<BE> + VecZnxNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let mk = TorusPrecision(ct.inner.base2k().0 * ct.inner.size() as u32);
    let rescale_bits = ct.log_delta.min(pt.log_delta);
    let off = ct.inner.max_k().as_usize();
    let (full_pt, scratch_rest) =
        offset_pt_from_scratch(module, ct.inner.n(), ct.inner.base2k(), mk, ct.inner.max_k(), pt, scratch);
    module.glwe_mul_plain_inplace(&mut ct.inner, off, &full_pt, scratch_rest);
    ct.log_delta += pt.log_delta;
    sign_extend_msb(ct);
    rescale(module, ct, rescale_bits, scratch_rest);
}

/// `res = ct * pt` using a pre-expanded plaintext (no allocation).
pub fn mul_prepared_pt<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    ct: &CKKSCiphertext<impl DataRef>,
    pt: &CKKSPlaintextPrepared<impl DataRef>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEMulPlain<BE> + VecZnxNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let size = ct.inner.size();
    let off = ct.inner.max_k().as_usize();
    module.glwe_mul_plain(&mut res.inner, off, &ct.inner, &pt.inner, scratch);
    res.log_delta = ct.log_delta + pt.log_delta;
    set_active_k_and_size(res, ct.inner.max_k().0, size);
    let rescale_bits = ct.log_delta.min(pt.log_delta);
    rescale(module, res, rescale_bits, scratch);
}

/// `ct *= pt` using a pre-expanded plaintext (no allocation).
pub fn mul_prepared_pt_inplace<BE: Backend>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertext<impl DataMut>,
    pt: &CKKSPlaintextPrepared<impl DataRef>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEMulPlain<BE> + VecZnxNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let rescale_bits = ct.log_delta.min(pt.log_delta);
    let off = ct.inner.max_k().as_usize();
    let actual_k = ct.inner.max_k();
    module.glwe_mul_plain_inplace(&mut ct.inner, off, &pt.inner, scratch);
    ct.log_delta += pt.log_delta;
    ct.inner.set_k(actual_k);
    sign_extend_msb(ct);
    rescale(module, ct, rescale_bits, scratch);
}

/// Returns the scratch bytes needed for [`mul_const`] and [`mul_const_inplace`].
pub fn mul_const_tmp_bytes<BE: Backend>(module: &Module<BE>, ct: &CKKSCiphertext<impl Data>) -> usize
where
    Module<BE>: GLWEMulPlain<BE> + VecZnxNormalizeTmpBytes,
{
    const_pt_scratch_bytes(ct) + mul_pt_tmp_bytes(module, ct)
}

/// Multiplies a ciphertext by a complex constant: `res = ct * (re + i*im)`.
pub fn mul_const<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    ct: &CKKSCiphertext<impl DataRef>,
    re: f64,
    im: f64,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEMulPlain<BE> + VecZnxNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let (pt, scratch_rest) = const_pt_from_scratch(ct, re, im, scratch);
    mul_pt(module, res, ct, &pt, scratch_rest);
}

/// Multiplies a ciphertext by a complex constant in place: `ct *= (re + i*im)`.
pub fn mul_const_inplace<BE: Backend>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertext<impl DataMut>,
    re: f64,
    im: f64,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEMulPlain<BE> + VecZnxNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let (pt, scratch_rest) = const_pt_from_scratch(ct, re, im, scratch);
    mul_pt_inplace(module, ct, &pt, scratch_rest);
}

/// Returns the scratch bytes needed for [`mul_int`].
pub fn mul_int_tmp_bytes<BE: Backend>(module: &Module<BE>, a: &CKKSCiphertext<impl Data>, b_size: usize) -> usize
where
    Module<BE>: GLWEMulConst<BE>,
{
    let off = a.inner.base2k().as_usize();
    module.glwe_mul_const_tmp_bytes(&a.inner, off, &a.inner, b_size)
}

/// Multiplies a ciphertext by a small integer: `res = ct * c`.
///
/// No rescale is performed; `log_delta` is unchanged.
pub fn mul_int<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    ct: &CKKSCiphertext<impl DataRef>,
    c: i64,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEMulConst<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let off = ct.inner.base2k().as_usize();
    module.glwe_mul_const(&mut res.inner, off, &ct.inner, &[c], scratch);
    res.log_delta = ct.log_delta;
}

/// Returns the scratch bytes needed for [`mul_int_inplace`].
pub fn mul_int_inplace_tmp_bytes<BE: Backend>(module: &Module<BE>, a: &CKKSCiphertext<impl Data>, b_size: usize) -> usize
where
    Module<BE>: GLWEMulConst<BE>,
{
    let off = a.inner.base2k().as_usize();
    module.glwe_mul_const_tmp_bytes(&a.inner, off, &a.inner, b_size)
}

/// Multiplies a ciphertext by a small integer in place: `ct *= c`.
///
/// No rescale is performed; `log_delta` is unchanged.
pub fn mul_int_inplace<BE: Backend>(module: &Module<BE>, ct: &mut CKKSCiphertext<impl DataMut>, c: i64, scratch: &mut Scratch<BE>)
where
    Module<BE>: GLWEMulConst<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let off = ct.inner.base2k().as_usize();
    module.glwe_mul_const_inplace(&mut ct.inner, off, &[c], scratch);
}
