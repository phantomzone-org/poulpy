//! CKKS leveled arithmetic: addition, subtraction, and negation.

use crate::layouts::{ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext};
use poulpy_core::{
    GLWEAdd, GLWESub,
    layouts::{Base2K, Degree, LWEInfos, TorusPrecision},
};
use poulpy_hal::{
    api::{ModuleN, VecZnxNegate, VecZnxNegateInplace},
    layouts::{Backend, DataMut, DataRef, Module},
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

/// Computes `res = ct + c`, adding the complex constant `c = re + i*im` to all slots.
pub fn add_const_ct<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    ct: &CKKSCiphertext<impl DataRef>,
    re: f64,
    im: f64,
) where
    Module<BE>: GLWEAdd + ModuleN,
{
    let pt = const_pt(module, ct, re, im);
    add_pt_ct(module, res, ct, &pt);
}

/// Computes `ct += c` in place, adding the complex constant `c = re + i*im` to all slots.
pub fn add_const_ct_inplace<BE: Backend>(module: &Module<BE>, ct: &mut CKKSCiphertext<impl DataMut>, re: f64, im: f64)
where
    Module<BE>: GLWEAdd + ModuleN,
{
    let pt = const_pt(module, ct, re, im);
    add_pt_ct_inplace(module, ct, &pt);
}

/// Computes `res = a - b`.
pub fn sub_ct_ct<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    a: &CKKSCiphertext<impl DataRef>,
    b: &CKKSCiphertext<impl DataRef>,
) where
    Module<BE>: GLWESub,
{
    assert_eq!(
        a.log_delta, b.log_delta,
        "sub_ct_ct: log_delta mismatch ({} != {})",
        a.log_delta, b.log_delta
    );
    res.log_delta = a.log_delta;
    module.glwe_sub(&mut res.inner, &a.inner, &b.inner);
}

/// Computes `res -= a` in place.
pub fn sub_ct_ct_inplace<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    a: &CKKSCiphertext<impl DataRef>,
) where
    Module<BE>: GLWESub,
{
    assert_eq!(
        res.log_delta, a.log_delta,
        "sub_ct_ct_inplace: log_delta mismatch ({} != {})",
        res.log_delta, a.log_delta
    );
    module.glwe_sub_inplace(&mut res.inner, &a.inner);
}

/// Computes `res = ct - pt`.
pub fn sub_pt_ct<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    ct: &CKKSCiphertext<impl DataRef>,
    pt: &CKKSPlaintext<impl DataRef>,
) where
    Module<BE>: GLWESub,
{
    assert_eq!(
        ct.log_delta, pt.log_delta,
        "sub_pt_ct: log_delta mismatch (ct={}, pt={})",
        ct.log_delta, pt.log_delta
    );
    res.log_delta = ct.log_delta;
    module.glwe_sub(&mut res.inner, &ct.inner, &pt.inner);
}

/// Computes `ct -= pt` in place.
pub fn sub_pt_ct_inplace<BE: Backend>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertext<impl DataMut>,
    pt: &CKKSPlaintext<impl DataRef>,
) where
    Module<BE>: GLWESub,
{
    assert_eq!(
        ct.log_delta, pt.log_delta,
        "sub_pt_ct_inplace: log_delta mismatch (ct={}, pt={})",
        ct.log_delta, pt.log_delta
    );
    module.glwe_sub_inplace(&mut ct.inner, &pt.inner);
}

/// Computes `res = ct - c`, subtracting the complex constant `c = re + i*im` from all slots.
pub fn sub_const_ct<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    ct: &CKKSCiphertext<impl DataRef>,
    re: f64,
    im: f64,
) where
    Module<BE>: GLWESub + ModuleN,
{
    let pt = const_pt(module, ct, re, im);
    sub_pt_ct(module, res, ct, &pt);
}

/// Computes `ct -= c` in place, subtracting the complex constant `c = re + i*im` from all slots.
pub fn sub_const_ct_inplace<BE: Backend>(module: &Module<BE>, ct: &mut CKKSCiphertext<impl DataMut>, re: f64, im: f64)
where
    Module<BE>: GLWESub + ModuleN,
{
    let pt = const_pt(module, ct, re, im);
    sub_pt_ct_inplace(module, ct, &pt);
}

/// Computes `res = -ct`.
pub fn neg_ct<BE: Backend>(module: &Module<BE>, res: &mut CKKSCiphertext<impl DataMut>, ct: &CKKSCiphertext<impl DataRef>)
where
    Module<BE>: VecZnxNegate,
{
    res.log_delta = ct.log_delta;
    let ncols = ct.inner.data().cols;
    for i in 0..ncols {
        module.vec_znx_negate(res.inner.data_mut(), i, ct.inner.data(), i);
    }
}

/// Computes `ct = -ct` in place.
pub fn neg_ct_inplace<BE: Backend>(module: &Module<BE>, ct: &mut CKKSCiphertext<impl DataMut>)
where
    Module<BE>: VecZnxNegateInplace,
{
    let ncols = ct.inner.data().cols;
    for i in 0..ncols {
        module.vec_znx_negate_inplace(ct.inner.data_mut(), i);
    }
}

/// Encodes the complex constant `re + i*im` as the polynomial `delta*re + delta*im*X^{N/2}`.
fn const_pt<BE: Backend>(module: &Module<BE>, ct: &CKKSCiphertext<impl DataRef>, re: f64, im: f64) -> CKKSPlaintext<Vec<u8>>
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
