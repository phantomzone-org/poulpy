use crate::reference::reim::{ReimArithmetic, reim_copy_ref, reim_zero_ref};

pub struct ReimArithmeticRef;

impl ReimArithmetic for ReimArithmeticRef {
    #[inline(always)]
    fn reim_add(res: &mut [f64], a: &[f64], b: &[f64]) {
        reim_add_ref(res, a, b);
    }

    #[inline(always)]
    fn reim_add_inplace(res: &mut [f64], a: &[f64]) {
        reim_add_inplace_ref(res, a);
    }

    #[inline(always)]
    fn reim_sub(res: &mut [f64], a: &[f64], b: &[f64]) {
        reim_sub_ref(res, a, b);
    }

    #[inline(always)]
    fn reim_sub_ab_inplace(res: &mut [f64], a: &[f64]) {
        reim_sub_ab_inplace_ref(res, a);
    }

    #[inline(always)]
    fn reim_sub_ba_inplace(res: &mut [f64], a: &[f64]) {
        reim_sub_ba_inplace_ref(res, a);
    }

    #[inline(always)]
    fn reim_negate(res: &mut [f64], a: &[f64]) {
        reim_negate_ref(res, a);
    }

    #[inline(always)]
    fn reim_negate_inplace(res: &mut [f64]) {
        reim_negate_inplace_ref(res);
    }

    #[inline(always)]
    fn reim_mul(res: &mut [f64], a: &[f64], b: &[f64]) {
        reim_mul_ref(res, a, b);
    }

    #[inline(always)]
    fn reim_mul_inplace(res: &mut [f64], a: &[f64]) {
        reim_mul_inplace_ref(res, a);
    }

    #[inline(always)]
    fn reim_addmul(res: &mut [f64], a: &[f64], b: &[f64]) {
        reim_addmul_ref(res, a, b);
    }

    #[inline(always)]
    fn reim_copy(res: &mut [f64], a: &[f64]) {
        reim_copy_ref(res, a);
    }

    #[inline(always)]
    fn reim_zero(res: &mut [f64]) {
        reim_zero_ref(res);
    }
}

#[inline(always)]
pub fn reim_add_ref(res: &mut [f64], a: &[f64], b: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
        assert_eq!(b.len(), res.len());
    }

    for i in 0..res.len() {
        res[i] = a[i] + b[i]
    }
}

#[inline(always)]
pub fn reim_add_inplace_ref(res: &mut [f64], a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
    }

    for i in 0..res.len() {
        res[i] += a[i]
    }
}

#[inline(always)]
pub fn reim_sub_ref(res: &mut [f64], a: &[f64], b: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
        assert_eq!(b.len(), res.len());
    }

    for i in 0..res.len() {
        res[i] = a[i] - b[i]
    }
}

#[inline(always)]
pub fn reim_sub_ab_inplace_ref(res: &mut [f64], a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
    }

    for i in 0..res.len() {
        res[i] -= a[i]
    }
}

#[inline(always)]
pub fn reim_sub_ba_inplace_ref(res: &mut [f64], a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
    }

    for i in 0..res.len() {
        res[i] = a[i] - res[i]
    }
}

#[inline(always)]
pub fn reim_negate_ref(res: &mut [f64], a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
    }

    for i in 0..res.len() {
        res[i] = -a[i]
    }
}

#[inline(always)]
pub fn reim_negate_inplace_ref(res: &mut [f64]) {
    for ri in res {
        *ri = -*ri
    }
}

#[inline(always)]
pub fn reim_addmul_ref(res: &mut [f64], a: &[f64], b: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
        assert_eq!(b.len(), res.len());
    }

    let m: usize = res.len() >> 1;

    let (rr, ri) = res.split_at_mut(m);
    let (ar, ai) = a.split_at(m);
    let (br, bi) = b.split_at(m);

    for i in 0..m {
        let _ar: f64 = ar[i];
        let _ai: f64 = ai[i];
        let _br: f64 = br[i];
        let _bi: f64 = bi[i];
        let _rr: f64 = _ar * _br - _ai * _bi;
        let _ri: f64 = _ar * _bi + _ai * _br;
        rr[i] += _rr;
        ri[i] += _ri;
    }
}

#[inline(always)]
pub fn reim_mul_inplace_ref(res: &mut [f64], a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
    }

    let m: usize = res.len() >> 1;

    let (rr, ri) = res.split_at_mut(m);
    let (ar, ai) = a.split_at(m);

    for i in 0..m {
        let _ar: f64 = ar[i];
        let _ai: f64 = ai[i];
        let _br: f64 = rr[i];
        let _bi: f64 = ri[i];
        let _rr: f64 = _ar * _br - _ai * _bi;
        let _ri: f64 = _ar * _bi + _ai * _br;
        rr[i] = _rr;
        ri[i] = _ri;
    }
}

#[inline(always)]
pub fn reim_mul_ref(res: &mut [f64], a: &[f64], b: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
        assert_eq!(b.len(), res.len());
    }

    let m: usize = res.len() >> 1;

    let (rr, ri) = res.split_at_mut(m);
    let (ar, ai) = a.split_at(m);
    let (br, bi) = b.split_at(m);

    for i in 0..m {
        let _ar: f64 = ar[i];
        let _ai: f64 = ai[i];
        let _br: f64 = br[i];
        let _bi: f64 = bi[i];
        let _rr: f64 = _ar * _br - _ai * _bi;
        let _ri: f64 = _ar * _bi + _ai * _br;
        rr[i] = _rr;
        ri[i] = _ri;
    }
}
