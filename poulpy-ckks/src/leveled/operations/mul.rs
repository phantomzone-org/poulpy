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
//! The output metadata law for ct × ct is:
//!
//! - `prefix_out  = min(prefix_a, prefix_b) - min(scale_a, scale_b)`
//! - `offset_out  = min(offset_a, offset_b) - min(scale_a, scale_b)`
//! - `scale_out   = max(scale_a, scale_b)`
//!
//! This corresponds to rescaling by `min(scale_a, scale_b)` bits, which
//! consumes the smaller scaling factor from the bivariate convolution product
//! by shrinking the active torus window.
//!
//! For ct × pt / ct × prepared-pt, multiplication is applied over the full
//! physical limb prefix of the ciphertext, but the product is re-tagged with
//! the source ciphertext's active `prefix_bits` / `offset_bits` before the
//! rescale step consumes visible precision.
//!
//! ## Variants
//!
//! | Function | Operands | Rescale |
//! |----------|----------|---------|
//! | [`square`] | ct × ct (same operand) | yes |
//! | [`mul`] / [`mul_aligned`] | ct × ct | yes |
//! | [`mul_pt`] / [`mul_pt_inplace`] | ct × compact pt | yes |
//! | [`mul_prepared_pt`] / [`mul_prepared_pt_inplace`] | ct × prepared pt | yes |
//! | [`mul_const`] / [`mul_const_inplace`] | ct × complex constant | yes |
//! | [`mul_int`] / [`mul_int_inplace`] | ct × integer | no |

use crate::{
    layouts::{
        ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext, plaintext_prepared::CKKSPlaintextPrepared, tensor::CKKSTensor,
    },
    leveled::operations::{
        align::assert_cts_aligned,
        level::rescale,
        utils::{const_pt_from_scratch, const_pt_scratch_bytes, offset_pt_from_scratch, sign_extend_msb},
    },
};
use poulpy_core::{
    GLWEMulConst, GLWEMulPlain, GLWEShift, GLWETensoring, ScratchTakeCore,
    layouts::{GLWEInfos, GLWELayout, GLWETensor, GLWETensorKeyPrepared, LWEInfos, Rank, SetGLWEInfos, TorusPrecision},
};
use poulpy_hal::{
    api::{VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

/// Sets the active precision window after multiplication or relinearization.
///
/// Updates `k`, shrinks `data.size` to the active limb count, zeros the
/// newly inactive tail, and sign-extends the MSB limb if `k` is not
/// limb-aligned. The post-condition is a valid [`CKKSCiphertext`] at the
/// given `k`, operationally indistinguishable from a freshly allocated one.
fn set_active_k_and_size(res: &mut CKKSCiphertext<impl DataMut>, k_eff: u32) {
    res.set_active_k(TorusPrecision(k_eff));
    res.zero_inactive_tail();
    sign_extend_msb(res);
    res.assert_valid("set_active_k_and_size result");
}

/// Returns the scratch bytes needed for [`tensor`] and [`tensor_aligned`].
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

/// Computes the tensor product of two already aligned CKKS ciphertexts.
pub fn tensor_aligned<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSTensor<impl DataMut>,
    a: &CKKSCiphertext<impl DataRef>,
    b: &CKKSCiphertext<impl DataRef>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWETensoring<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    a.assert_valid("tensor lhs");
    b.assert_valid("tensor rhs");
    assert_cts_aligned(a, b, "tensor_aligned");
    tensor(module, res, a, b, scratch);
}

/// Computes the tensor product of two CKKS ciphertexts.
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
    a.assert_valid("tensor lhs");
    b.assert_valid("tensor rhs");
    assert_eq!(a.inner.base2k(), b.inner.base2k(), "tensor: base2k mismatch");
    let off = a.inner.max_k().as_usize().max(b.inner.max_k().as_usize());
    res.torus_scale_bits = a.torus_scale_bits() + b.torus_scale_bits();
    res.offset_bits = a.offset_bits().min(b.offset_bits());
    module.glwe_tensor_apply(&mut res.inner, off, &a.inner, &b.inner, scratch);
    res.assert_valid("tensor result");
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
/// After relinearization, retracts the active precision window to `k_eff`,
/// zeros the newly inactive tail, and sign-extends the MSB limb.
pub fn relinearize<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    tensor: &CKKSTensor<impl DataRef>,
    k_eff: u32,
    tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWETensoring<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    tensor.assert_valid("relinearize input tensor");
    module.glwe_tensor_relinearize(&mut res.inner, &tensor.inner, tsk, tsk.size(), scratch);
    res.torus_scale_bits = tensor.torus_scale_bits;
    res.offset_bits = tensor.offset_bits;
    set_active_k_and_size(res, k_eff);
}

/// Tensor buffer size for `mul(a, b)` expressed in limbs of the operands'
/// `base2k`.
///
/// Matches the meaningful output count of [`GLWETensoring::glwe_tensor_apply`]:
/// `a.size + b.size - offset_limbs`, where `offset_limbs` is
/// `max(k_a, k_b) / base2k` (integer). This is the minimum tensor width
/// that preserves all limbs the subsequent `rescale` needs. It scales
/// linearly with the **active** operand size, so shrunk ciphertexts produce
/// proportionally smaller tensors (and cheaper muls).
fn mul_tensor_size(a_size: usize, b_size: usize, a_k_bits: u32, b_k_bits: u32, base2k: u32) -> usize {
    let offset_limbs = (a_k_bits.max(b_k_bits) / base2k) as usize;
    // a_size + b_size - offset_limbs is always >= 1 for valid inputs because
    // `offset_limbs <= max(a_size, b_size)` and `a_size + b_size - max >= 1`.
    a_size + b_size - offset_limbs
}

/// Returns the scratch bytes needed for [`mul`] and [`mul_aligned`].
///
/// Scales with the **active** `size()` of the operands (not their
/// underlying `max_size()`), via [`mul_tensor_size`], so shrunk ciphertexts
/// consume strictly less scratch than fresh ones at a higher level.
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
    let tensor_size = mul_tensor_size(
        a.inner.size(),
        b.inner.size(),
        a.inner.k().0,
        b.inner.k().0,
        a.inner.base2k().0,
    );
    let mut layout = a.inner.glwe_layout();
    layout.k = TorusPrecision(a.inner.base2k().0 * tensor_size as u32);
    let tensor_bytes = GLWETensor::bytes_of(a.inner.n(), a.inner.base2k(), layout.k, Rank(1));
    let op_bytes = module
        .glwe_tensor_apply_tmp_bytes(&layout, off, &a.inner, &b.inner)
        .max(module.glwe_tensor_relinearize_tmp_bytes(&layout, &layout, tsk));
    tensor_bytes + op_bytes
}

/// Returns the scratch bytes needed for [`square`].
///
/// Scales with the **active** `size()` of the operand (not its `max_size()`).
pub fn square_tmp_bytes<BE: Backend>(
    module: &Module<BE>,
    a: &CKKSCiphertext<impl Data>,
    tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
) -> usize
where
    Module<BE>: GLWETensoring<BE>,
{
    let off = a.inner.max_k().as_usize();
    let tensor_size = mul_tensor_size(
        a.inner.size(),
        a.inner.size(),
        a.inner.k().0,
        a.inner.k().0,
        a.inner.base2k().0,
    );
    let mut layout = a.inner.glwe_layout();
    layout.k = TorusPrecision(a.inner.base2k().0 * tensor_size as u32);
    let tensor_bytes = GLWETensor::bytes_of(a.inner.n(), a.inner.base2k(), layout.k, Rank(1));
    let op_bytes = module
        .glwe_tensor_square_apply_tmp_bytes(&layout, off, &a.inner)
        .max(module.glwe_tensor_relinearize_tmp_bytes(&layout, &layout, tsk));
    tensor_bytes + op_bytes
}

/// Multiplies two already-aligned CKKS ciphertexts: `res = a * b`.
///
/// Both operands must have the same `offset_bits` and `prefix_bits`.
/// Runs the full tensor → relinearize → rescale pipeline. The rescale
/// consumes `min(torus_scale_bits_a, torus_scale_bits_b)` bits.
pub fn mul_aligned<BE: Backend>(
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
    a.assert_valid("mul lhs");
    b.assert_valid("mul rhs");
    assert_cts_aligned(a, b, "mul_aligned");
    let rescale_bits = a.torus_scale_bits().min(b.torus_scale_bits());
    let active_prefix = a.delta().min(b.delta());
    let base2k = a.inner.base2k().0;
    let tensor_size =
        mul_tensor_size(a.inner.size(), b.inner.size(), a.inner.k().0, b.inner.k().0, base2k).min(res.inner.max_size());
    let tensor_k = TorusPrecision(base2k * tensor_size as u32);
    let tensor_layout = GLWELayout {
        n: a.inner.n(),
        base2k: a.inner.base2k(),
        k: tensor_k,
        rank: Rank(1),
    };
    let (tensor_inner, scratch_rest) = scratch.take_glwe_tensor(&tensor_layout);
    let mut tensor_res = CKKSTensor {
        inner: tensor_inner,
        offset_bits: 0,
        torus_scale_bits: 0,
    };
    tensor_aligned(module, &mut tensor_res, a, b, scratch_rest);
    // Widen `res` to the tensor window; relinearize then retracts it back to
    // `active_prefix` and re-zeroes the newly inactive tail.
    res.set_active_k(tensor_k);
    relinearize(module, res, &tensor_res, active_prefix, tsk, scratch_rest);
    rescale(module, res, rescale_bits, scratch_rest);
    res.assert_valid("mul_aligned result");
}

/// Multiplies two CKKS ciphertexts: `res = a * b`.
///
/// Runs the full tensor → relinearize → rescale pipeline. The rescale
/// consumes `min(torus_scale_bits_a, torus_scale_bits_b)` bits and leaves the
/// result with active precision `min(k_a, k_b) - rescale_bits`.
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
    a.assert_valid("mul lhs");
    b.assert_valid("mul rhs");
    let rescale_bits = a.torus_scale_bits().min(b.torus_scale_bits());
    let active_prefix = a.delta().min(b.delta());
    let base2k = a.inner.base2k().0;
    let tensor_size =
        mul_tensor_size(a.inner.size(), b.inner.size(), a.inner.k().0, b.inner.k().0, base2k).min(res.inner.max_size());
    let tensor_k = TorusPrecision(base2k * tensor_size as u32);
    let tensor_layout = GLWELayout {
        n: a.inner.n(),
        base2k: a.inner.base2k(),
        k: tensor_k,
        rank: Rank(1),
    };
    let (tensor_inner, scratch_rest) = scratch.take_glwe_tensor(&tensor_layout);
    let mut tensor_res = CKKSTensor {
        inner: tensor_inner,
        offset_bits: 0,
        torus_scale_bits: 0,
    };
    tensor(module, &mut tensor_res, a, b, scratch_rest);
    // See `mul_aligned` for the widen/retract rationale.
    res.set_active_k(tensor_k);
    relinearize(module, res, &tensor_res, active_prefix, tsk, scratch_rest);
    rescale(module, res, rescale_bits, scratch_rest);
    res.assert_valid("mul result");
}

/// Squares a CKKS ciphertext: `res = a * a`.
///
/// Runs the full tensor -> relinearize -> rescale pipeline through the
/// dedicated self-convolution entry point in `poulpy-core`.
pub fn square<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    a: &CKKSCiphertext<impl DataRef>,
    tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWETensoring<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    a.assert_valid("square input");
    let rescale_bits = a.torus_scale_bits();
    let active_prefix = a.delta();
    let base2k = a.inner.base2k().0;
    let tensor_size =
        mul_tensor_size(a.inner.size(), a.inner.size(), a.inner.k().0, a.inner.k().0, base2k).min(res.inner.max_size());
    let tensor_k = TorusPrecision(base2k * tensor_size as u32);
    let tensor_layout = GLWELayout {
        n: a.inner.n(),
        base2k: a.inner.base2k(),
        k: tensor_k,
        rank: Rank(1),
    };
    let (tensor_inner, scratch_rest) = scratch.take_glwe_tensor(&tensor_layout);
    let mut tensor_res = CKKSTensor {
        inner: tensor_inner,
        offset_bits: a.offset_bits(),
        torus_scale_bits: a.torus_scale_bits() * 2,
    };
    let off = a.inner.max_k().as_usize();
    module.glwe_tensor_square_apply(&mut tensor_res.inner, off, &a.inner, scratch_rest);
    tensor_res.assert_valid("square tensor result");
    // See `mul_aligned` for the widen/retract rationale.
    res.set_active_k(tensor_k);
    relinearize(module, res, &tensor_res, active_prefix, tsk, scratch_rest);
    rescale(module, res, rescale_bits, scratch_rest);
    res.assert_valid("square result");
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
/// Offset is `k_ct`. Rescale consumes `min(torus_scale_bits_ct, embed_bits_pt)` bits.
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
    ct.assert_valid("mul_pt input");
    let rescale_bits = ct.torus_scale_bits().min(pt.embed_bits());
    let product_scale_bits = ct.torus_scale_bits() + pt.embed_bits();
    let mk = TorusPrecision(ct.inner.base2k().0 * ct.inner.size() as u32);
    let product_offset_bits = ct.offset_bits();
    let active_prefix_bits = ct.delta();
    let off = mk.as_usize();
    let (full_pt, scratch_rest) = offset_pt_from_scratch(module, ct.inner.n(), ct.inner.base2k(), mk, mk, pt, scratch);
    module.glwe_mul_plain(&mut res.inner, off, &ct.inner, &full_pt, scratch_rest);
    res.torus_scale_bits = product_scale_bits;
    res.offset_bits = product_offset_bits;
    set_active_k_and_size(res, active_prefix_bits);
    rescale(module, res, rescale_bits, scratch_rest);
    res.assert_valid("mul_pt result");
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
    ct.assert_valid("mul_pt_inplace input");
    let rescale_bits = ct.torus_scale_bits().min(pt.embed_bits());
    let product_scale_bits = ct.torus_scale_bits() + pt.embed_bits();
    let mk = TorusPrecision(ct.inner.base2k().0 * ct.inner.size() as u32);
    let product_offset_bits = ct.offset_bits();
    let actual_k = ct.inner.k();
    let off = mk.as_usize();
    let (full_pt, scratch_rest) = offset_pt_from_scratch(module, ct.inner.n(), ct.inner.base2k(), mk, mk, pt, scratch);
    ct.inner.set_k(mk);
    module.glwe_mul_plain_inplace(&mut ct.inner, off, &full_pt, scratch_rest);
    ct.inner.set_k(actual_k);
    ct.torus_scale_bits = product_scale_bits;
    ct.offset_bits = product_offset_bits;
    sign_extend_msb(ct);
    rescale(module, ct, rescale_bits, scratch_rest);
    ct.assert_valid("mul_pt_inplace result");
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
    ct.assert_valid("mul_prepared_pt input");
    let rescale_bits = ct.torus_scale_bits().min(pt.embed_bits());
    let product_scale_bits = ct.torus_scale_bits() + pt.embed_bits();
    let mk = TorusPrecision(ct.inner.base2k().0 * ct.inner.size() as u32);
    let product_offset_bits = ct.offset_bits();
    let active_prefix_bits = ct.delta();
    let off = mk.as_usize();
    module.glwe_mul_plain(&mut res.inner, off, &ct.inner, &pt.inner, scratch);
    res.torus_scale_bits = product_scale_bits;
    res.offset_bits = product_offset_bits;
    set_active_k_and_size(res, active_prefix_bits);
    rescale(module, res, rescale_bits, scratch);
    res.assert_valid("mul_prepared_pt result");
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
    ct.assert_valid("mul_prepared_pt_inplace input");
    let rescale_bits = ct.torus_scale_bits().min(pt.embed_bits());
    let product_scale_bits = ct.torus_scale_bits() + pt.embed_bits();
    let mk = TorusPrecision(ct.inner.base2k().0 * ct.inner.size() as u32);
    let product_offset_bits = ct.offset_bits();
    let actual_k = ct.inner.k();
    let off = mk.as_usize();
    ct.inner.set_k(mk);
    module.glwe_mul_plain_inplace(&mut ct.inner, off, &pt.inner, scratch);
    ct.inner.set_k(actual_k);
    ct.torus_scale_bits = product_scale_bits;
    ct.offset_bits = product_offset_bits;
    sign_extend_msb(ct);
    rescale(module, ct, rescale_bits, scratch);
    ct.assert_valid("mul_prepared_pt_inplace result");
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
    Module<BE>: GLWEMulPlain<BE> + VecZnxNormalize<BE> + GLWEShift<BE>,
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
    Module<BE>: GLWEMulPlain<BE> + VecZnxNormalize<BE> + GLWEShift<BE>,
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
/// No rescale is performed; `torus_scale_bits` is unchanged.
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
    ct.assert_valid("mul_int input");
    let off = ct.inner.base2k().as_usize();
    module.glwe_mul_const(&mut res.inner, off, &ct.inner, &[c], scratch);
    res.torus_scale_bits = ct.torus_scale_bits();
    res.offset_bits = ct.offset_bits();
    res.assert_valid("mul_int result");
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
/// No rescale is performed; `torus_scale_bits` is unchanged.
pub fn mul_int_inplace<BE: Backend>(module: &Module<BE>, ct: &mut CKKSCiphertext<impl DataMut>, c: i64, scratch: &mut Scratch<BE>)
where
    Module<BE>: GLWEMulConst<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    ct.assert_valid("mul_int_inplace input");
    let off = ct.inner.base2k().as_usize();
    module.glwe_mul_const_inplace(&mut ct.inner, off, &[c], scratch);
    ct.assert_valid("mul_int_inplace result");
}
