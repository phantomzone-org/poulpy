#[inline(always)]
pub fn reim_from_znx_i64_ref(res: &mut [f64], a: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len())
    }

    for i in 0..res.len() {
        res[i] = a[i] as f64
    }
}

#[inline(always)]
pub fn reim_to_znx_i64_ref(res: &mut [i64], divisor: f64, a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len())
    }
    let inv_div = 1. / divisor;
    for i in 0..res.len() {
        res[i] = (a[i] * inv_div).round() as i64
    }
}

#[inline(always)]
pub fn reim_to_znx_i64_inplace_ref(res: &mut [f64], divisor: f64) {
    let inv_div = 1. / divisor;
    for ri in res {
        *ri = f64::from_bits(((*ri * inv_div).round() as i64) as u64)
    }
}
