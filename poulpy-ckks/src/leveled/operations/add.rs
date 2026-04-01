//! CKKS ciphertext addition.

use super::utils::const_pt;
use crate::layouts::{ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext};
use poulpy_core::GLWEAdd;
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, DataMut, DataRef, Module},
};

/// Computes `res = a + b`.
pub fn add<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    a: &CKKSCiphertext<impl DataRef>,
    b: &CKKSCiphertext<impl DataRef>,
) where
    Module<BE>: GLWEAdd,
{
    assert_eq!(
        a.log_delta, b.log_delta,
        "add: log_delta mismatch ({} != {})",
        a.log_delta, b.log_delta
    );
    res.log_delta = a.log_delta;
    module.glwe_add(&mut res.inner, &a.inner, &b.inner);
}

/// Computes `res += a` in place.
pub fn add_inplace<BE: Backend>(module: &Module<BE>, res: &mut CKKSCiphertext<impl DataMut>, a: &CKKSCiphertext<impl DataRef>)
where
    Module<BE>: GLWEAdd,
{
    assert_eq!(
        res.log_delta, a.log_delta,
        "add_inplace: log_delta mismatch ({} != {})",
        res.log_delta, a.log_delta
    );
    module.glwe_add_inplace(&mut res.inner, &a.inner);
}

/// Computes `res = ct + pt`.
pub fn add_pt<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    ct: &CKKSCiphertext<impl DataRef>,
    pt: &CKKSPlaintext<impl DataRef>,
) where
    Module<BE>: GLWEAdd,
{
    assert_eq!(
        ct.log_delta, pt.log_delta,
        "add_pt: log_delta mismatch (ct={}, pt={})",
        ct.log_delta, pt.log_delta
    );
    res.log_delta = ct.log_delta;
    module.glwe_add(&mut res.inner, &ct.inner, &pt.inner);
}

/// Computes `ct += pt` in place.
pub fn add_pt_inplace<BE: Backend>(module: &Module<BE>, ct: &mut CKKSCiphertext<impl DataMut>, pt: &CKKSPlaintext<impl DataRef>)
where
    Module<BE>: GLWEAdd,
{
    assert_eq!(
        ct.log_delta, pt.log_delta,
        "add_pt_inplace: log_delta mismatch (ct={}, pt={})",
        ct.log_delta, pt.log_delta
    );
    module.glwe_add_inplace(&mut ct.inner, &pt.inner);
}

/// Computes `res = ct + c` where `c = re + i*im`.
pub fn add_const<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    ct: &CKKSCiphertext<impl DataRef>,
    re: f64,
    im: f64,
) where
    Module<BE>: GLWEAdd + ModuleN,
{
    let pt = const_pt(module, ct, re, im);
    add_pt(module, res, ct, &pt);
}

/// Computes `ct += c` in place where `c = re + i*im`.
pub fn add_const_inplace<BE: Backend>(module: &Module<BE>, ct: &mut CKKSCiphertext<impl DataMut>, re: f64, im: f64)
where
    Module<BE>: GLWEAdd + ModuleN,
{
    let pt = const_pt(module, ct, re, im);
    add_pt_inplace(module, ct, &pt);
}
