//! Shared helpers for CKKS leveled operations.

use crate::layouts::{ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext};
use poulpy_core::layouts::{Base2K, Degree, LWEInfos, TorusPrecision};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, DataRef, Module},
};

/// Encodes the complex constant `re + i*im` as the polynomial `delta*re + delta*im*X^{N/2}`.
pub(crate) fn const_pt<BE: Backend>(
    module: &Module<BE>,
    ct: &CKKSCiphertext<impl DataRef>,
    re: f64,
    im: f64,
) -> CKKSPlaintext<Vec<u8>>
where
    Module<BE>: ModuleN,
{
    let n = module.n();
    let delta = (2.0f64).powi(ct.log_delta as i32);
    let base2k = ct.inner.base2k().0 as usize;
    let log_delta = ct.log_delta as usize;
    let v_re = (delta * re).round() as i64;
    let v_im = (delta * im).round() as i64;

    let mut pt = CKKSPlaintext::alloc(
        Degree(n as u32),
        Base2K(ct.inner.base2k().0),
        TorusPrecision(ct.inner.k().0),
        ct.log_delta,
    );
    pt.inner.data.encode_coeff_i64(base2k, 0, log_delta, 0, v_re);
    pt.inner.data.encode_coeff_i64(base2k, 0, log_delta, n / 2, v_im);
    pt
}
