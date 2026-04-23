//! CKKS composite operations built from the primitive operation modules.
//!
//! ## Operations
//!
//! | Module | Operation |
//! |--------|-----------|
//! | [`add_many`] | Sum of many ciphertexts with deferred normalization |
//! | [`mul_many`] | Product of many ciphertexts through a balanced multiplication tree |
//! | [`mul_add`] | Fused multiply-add variants (`dst += a · b`) for product accumulation with scratch sizing and CKKS alignment handled by the API |
//! | [`mul_sub`] | Fused multiply-sub variants (`dst -= a · b`) for residual/update steps with scratch sizing and CKKS alignment handled by the API |
//! | [`dot_product`] | Inner products over ciphertext, plaintext-vector, and constant weights |

pub mod add_many;
pub mod dot_product;
pub mod mul_add;
pub mod mul_many;
pub mod mul_sub;

pub use add_many::CKKSAddManyOps;
pub use dot_product::CKKSDotProductOps;
pub use mul_add::CKKSMulAddOps;
pub use mul_many::CKKSMulManyOps;
pub use mul_sub::CKKSMulSubOps;

use anyhow::{Result, ensure};
use poulpy_core::{
    ScratchTakeCore,
    layouts::{GLWEInfos, LWEInfos},
};
use poulpy_hal::layouts::{Backend, Data, DataMut, Scratch};

use crate::{CKKSMeta, layouts::CKKSCiphertext};

/// Take a scratch-backed CKKS ciphertext sized like `dst`, for use as a
/// temporary product buffer in fused multiply-add / multiply-sub paths.
pub(crate) fn take_mul_tmp<'a, BE: Backend, D: DataMut>(
    dst: &CKKSCiphertext<D>,
    scratch: &'a mut Scratch<BE>,
) -> (CKKSCiphertext<&'a mut [u8]>, &'a mut Scratch<BE>)
where
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let layout = dst.glwe_layout();
    let (tmp, scratch_r) = scratch.take_glwe(&layout);
    (CKKSCiphertext::from_inner(tmp, CKKSMeta::default()), scratch_r)
}

/// Guard against i64 overflow when `n` K-normalized summands are accumulated
/// before a single trailing normalize. See §3.3 of
/// [eprint 2023/771](https://eprint.iacr.org/2023/771).
pub(crate) fn ensure_accumulation_fits<D: Data>(op: &'static str, dst: &CKKSCiphertext<D>, n: usize) -> Result<()> {
    let base2k: usize = dst.base2k().as_usize();
    ensure!(base2k < 64, "{op}: unsupported base2k={base2k}");
    ensure!(
        n <= (1usize << (63 - base2k)),
        "{op}: {n} terms risks i64 overflow at base2k={base2k}",
    );
    Ok(())
}
