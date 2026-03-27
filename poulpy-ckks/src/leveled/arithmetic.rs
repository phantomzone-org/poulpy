//! CKKS leveled arithmetic: addition and its variants.

use crate::layouts::{ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext};
use poulpy_core::{
    GLWEAdd,
    layouts::{Base2K, Degree, LWEInfos, TorusPrecision},
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, DataMut, DataRef, Module, ZnxViewMut},
};

/// Computes `res = a + b`.
pub fn add_ct_ct<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    a: &CKKSCiphertext<impl DataRef>,
    b: &CKKSCiphertext<impl DataRef>,
) where
    Module<BE>: GLWEAdd,
{
    assert_eq!(
        a.log_delta, b.log_delta,
        "add_ct_ct: log_delta mismatch ({} != {})",
        a.log_delta, b.log_delta
    );
    res.log_delta = a.log_delta;
    module.glwe_add(&mut res.inner, &a.inner, &b.inner);
}

/// Computes `res += a` in place.
pub fn add_ct_ct_inplace<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    a: &CKKSCiphertext<impl DataRef>,
) where
    Module<BE>: GLWEAdd,
{
    assert_eq!(
        res.log_delta, a.log_delta,
        "add_ct_ct_inplace: log_delta mismatch ({} != {})",
        res.log_delta, a.log_delta
    );
    module.glwe_add_inplace(&mut res.inner, &a.inner);
}

/// Computes `res = ct + pt`.
pub fn add_pt_ct<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    ct: &CKKSCiphertext<impl DataRef>,
    pt: &CKKSPlaintext<impl DataRef>,
) where
    Module<BE>: GLWEAdd,
{
    assert_eq!(
        ct.log_delta, pt.log_delta,
        "add_pt_ct: log_delta mismatch (ct={}, pt={})",
        ct.log_delta, pt.log_delta
    );
    res.log_delta = ct.log_delta;
    module.glwe_add(&mut res.inner, &ct.inner, &pt.inner);
}

/// Computes `ct += pt` in place.
pub fn add_pt_ct_inplace<BE: Backend>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertext<impl DataMut>,
    pt: &CKKSPlaintext<impl DataRef>,
) where
    Module<BE>: GLWEAdd,
{
    assert_eq!(
        ct.log_delta, pt.log_delta,
        "add_pt_ct_inplace: log_delta mismatch (ct={}, pt={})",
        ct.log_delta, pt.log_delta
    );
    module.glwe_add_inplace(&mut ct.inner, &pt.inner);
}

/// Computes `res = ct + (re + i*im)`, adding the scalar to all slots.
pub fn add_cleartext_ct<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    ct: &CKKSCiphertext<impl DataRef>,
    re: f64,
    im: f64,
) where
    Module<BE>: GLWEAdd + ModuleN,
{
    let pt = scalar_pt(module, ct, re, im);
    add_pt_ct(module, res, ct, &pt);
}

/// Computes `ct += (re + i*im)` in place, adding the scalar to all slots.
pub fn add_cleartext_ct_inplace<BE: Backend>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertext<impl DataMut>,
    re: f64,
    im: f64,
) where
    Module<BE>: GLWEAdd + ModuleN,
{
    let pt = scalar_pt(module, ct, re, im);
    add_pt_ct_inplace(module, ct, &pt);
}

/// Encodes a complex scalar into a plaintext polynomial delta*re + delta*im*X^{N/2}.
fn scalar_pt<BE: Backend>(
    module: &Module<BE>,
    ct: &CKKSCiphertext<impl DataRef>,
    re: f64,
    im: f64,
) -> CKKSPlaintext<Vec<u8>>
where
    Module<BE>: ModuleN,
{
    let n = module.n();
    let base2k = ct.inner.base2k().0 as usize;
    let log_delta = ct.log_delta as usize;

    assert!(
        log_delta <= base2k,
        "scalar_pt: log_delta ({}) > base2k ({})",
        log_delta,
        base2k
    );

    let delta = (2.0f64).powi(ct.log_delta as i32);
    let res_offset = (base2k - log_delta) as u32;
    let v_re = (delta * re).round() as i64;
    let v_im = (delta * im).round() as i64;

    let mut pt = CKKSPlaintext::alloc(
        Degree(n as u32),
        Base2K(ct.inner.base2k().0),
        TorusPrecision(ct.inner.k().0),
        ct.log_delta,
    );
    pt.inner.data.at_mut(0, 0)[0] = v_re << res_offset;
    pt.inner.data.at_mut(0, 0)[n / 2] = v_im << res_offset;
    pt
}
