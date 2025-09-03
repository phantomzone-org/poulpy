use crate::reference::znx::znx_negate_i64_ref;

pub fn znx_rotate_i64_ref(p: i64, res: &mut [i64], src: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), src.len());
    }

    let n: usize = res.len();

    let mp_2n: usize = ((-p) & (2 * n as i64 - 1)) as usize; // -p % 2n
    let mp_1n: usize = mp_2n & (n - 1); // -p % n
    let mp_1n_neg: usize = n - mp_1n; //  p % n
    let neg_first: bool = mp_1n < n;

    let (dst1, dst2) = res.split_at_mut(mp_1n);
    let (src1, src2) = src.split_at(mp_1n_neg);

    if neg_first {
        dst1.copy_from_slice(src2);
        znx_negate_i64_ref(dst2, src1);
    } else {
        znx_negate_i64_ref(dst1, src2);
        dst2.copy_from_slice(src1);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub fn znx_rotate_i64_avx(p: i64, res: &mut [i64], src: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), src.len());
    }

    use crate::reference::znx::znx_negate_i64_avx;

    let n: usize = res.len();

    let mp_2n: usize = ((-p) & (2 * n as i64 - 1)) as usize; // -p % 2n
    let mp_1n: usize = mp_2n & (n - 1); // -p % n
    let mp_1n_neg: usize = n - mp_1n; //  p % n
    let neg_first: bool = mp_1n < n;

    let (dst1, dst2) = res.split_at_mut(mp_1n);
    let (src1, src2) = src.split_at(mp_1n_neg);

    #[allow(unused_unsafe)]
    unsafe {
        if neg_first {
            dst1.copy_from_slice(src2);
            znx_negate_i64_avx(dst2, src1);
        } else {
            znx_negate_i64_avx(dst1, src2);
            dst2.copy_from_slice(src1);
        }
    }
}
