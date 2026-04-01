//! CKKS ciphertext-ciphertext multiplication.

use crate::{
    layouts::{ciphertext::CKKSCiphertext, keys::tensor_key_prepared::CKKSTensorKeyPrepared, tensor::CKKSTensor},
    leveled::operations::level::rescale,
};
use poulpy_core::{GLWEShift, GLWETensoring, ScratchTakeCore, layouts::LWEInfos};
use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch};

/// Returns `res_offset` for the CKKS tensor product.
fn ckks_res_offset(base2k: usize, log_delta_a: usize, log_delta_b: usize) -> usize {
    2 * base2k - log_delta_a - log_delta_b
}

/// Returns scratch bytes needed by [`tensor`].
pub fn tensor_tmp_bytes<BE: Backend>(
    module: &Module<BE>,
    res: &CKKSTensor<impl Data>,
    a: &CKKSCiphertext<impl Data>,
    b: &CKKSCiphertext<impl Data>,
) -> usize
where
    Module<BE>: GLWETensoring<BE>,
{
    let res_offset = ckks_res_offset(a.inner.base2k().as_usize(), a.log_delta as usize, b.log_delta as usize);
    module.glwe_tensor_apply_tmp_bytes(&res.inner, res_offset, &a.inner, &b.inner)
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
    assert_eq!(a.inner.base2k(), b.inner.base2k(), "tensor: base2k mismatch");

    let res_offset = ckks_res_offset(a.inner.base2k().as_usize(), a.log_delta as usize, b.log_delta as usize);
    res.log_delta = a.log_delta + b.log_delta;
    module.glwe_tensor_apply(&mut res.inner, res_offset, &a.inner, &b.inner, scratch);
}

/// Returns scratch bytes needed by [`relinearize`].
pub fn relinearize_tmp_bytes<BE: Backend>(
    module: &Module<BE>,
    res: &CKKSCiphertext<impl Data>,
    tensor: &CKKSTensor<impl Data>,
    tsk: &CKKSTensorKeyPrepared<impl DataRef, BE>,
) -> usize
where
    Module<BE>: GLWETensoring<BE>,
{
    module.glwe_tensor_relinearize_tmp_bytes(&res.inner, &tensor.inner, &tsk.inner)
}

/// Relinearizes a tensor product back to a standard CKKS ciphertext.
pub fn relinearize<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    tensor: &CKKSTensor<impl DataRef>,
    tsk: &CKKSTensorKeyPrepared<impl DataRef, BE>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWETensoring<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    module.glwe_tensor_relinearize(&mut res.inner, &tensor.inner, &tsk.inner, tsk.size(), scratch);
    res.log_delta = tensor.log_delta;
}

/// Returns scratch bytes needed by [`mul`].
pub fn mul_tmp_bytes<BE: Backend>(
    module: &Module<BE>,
    a: &CKKSCiphertext<impl Data>,
    b: &CKKSCiphertext<impl Data>,
    tsk: &CKKSTensorKeyPrepared<impl DataRef, BE>,
) -> usize
where
    Module<BE>: GLWETensoring<BE> + GLWEShift<BE>,
{
    let res_offset = ckks_res_offset(a.inner.base2k().as_usize(), a.log_delta as usize, b.log_delta as usize);

    let tensor = CKKSTensor::alloc_from_infos(&a.inner, 0);
    let res = CKKSCiphertext::alloc(a.inner.n(), a.inner.base2k(), a.inner.k(), a.log_delta);

    module
        .glwe_tensor_apply_tmp_bytes(&tensor.inner, res_offset, &a.inner, &b.inner)
        .max(module.glwe_tensor_relinearize_tmp_bytes(&res.inner, &tensor.inner, &tsk.inner))
        .max(module.glwe_rsh_tmp_byte())
}

/// Computes `res = a * b` (tensor + relinearize + rescale).
pub fn mul<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    a: &CKKSCiphertext<impl DataRef>,
    b: &CKKSCiphertext<impl DataRef>,
    tsk: &CKKSTensorKeyPrepared<impl DataRef, BE>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWETensoring<BE> + GLWEShift<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let rescale_bits = a.log_delta.min(b.log_delta);
    let mut tensor_res = CKKSTensor::alloc_from_infos(&a.inner, 0);
    tensor(module, &mut tensor_res, a, b, scratch);
    relinearize(module, res, &tensor_res, tsk, scratch);
    rescale(module, res, rescale_bits, scratch);
}
