use itertools::izip;

use crate::reference::znx::ZnxNormalize;

pub struct ZnxNormalizeRef;

impl ZnxNormalize for ZnxNormalizeRef {
    #[inline(always)]
    fn znx_normalize_final_step(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_final_step_ref(basek, lsh, x, a, carry);
    }

    #[inline(always)]
    fn znx_normalize_final_step_inplace(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_final_step_inplace_ref(basek, lsh, x, carry);
    }

    #[inline(always)]
    fn znx_normalize_first_step(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_ref(basek, lsh, x, a, carry);
    }

    #[inline(always)]
    fn znx_normalize_first_step_carry_only(basek: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_carry_only_ref(basek, lsh, x, carry);
    }

    #[inline(always)]
    fn znx_normalize_first_step_inplace(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_first_step_inplace_ref(basek, lsh, x, carry);
    }

    #[inline(always)]
    fn znx_normalize_middle_step(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_ref(basek, lsh, x, a, carry);
    }

    #[inline(always)]
    fn znx_normalize_middle_step_carry_only(basek: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_carry_only_ref(basek, lsh, x, carry);
    }

    #[inline(always)]
    fn znx_normalize_middle_step_inplace(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_middle_step_inplace_ref(basek, lsh, x, carry);
    }
}

#[inline(always)]
pub(crate) fn get_digit(basek: usize, x: i64) -> i64 {
    (x << (u64::BITS - basek as u32)) >> (u64::BITS - basek as u32)
}

#[inline(always)]
pub(crate) fn get_carry(basek: usize, x: i64, digit: i64) -> i64 {
    (x.wrapping_sub(digit)) >> basek
}

#[inline(always)]
pub(crate) fn znx_normalize_first_step_carry_only_ref(basek: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
        assert!(lsh < basek);
    }

    if lsh == 0 {
        x.iter().zip(carry.iter_mut()).for_each(|(x, c)| {
            *c = get_carry(basek, *x, get_digit(basek, *x));
        });
    } else {
        let basek_lsh: usize = basek - lsh;
        x.iter().zip(carry.iter_mut()).for_each(|(x, c)| {
            *c = get_carry(basek_lsh, *x, get_digit(basek_lsh, *x));
        });
    }
}

#[inline(always)]
pub(crate) fn znx_normalize_first_step_inplace_ref(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
        assert!(lsh < basek);
    }

    if lsh == 0 {
        x.iter_mut().zip(carry.iter_mut()).for_each(|(x, c)| {
            let digit: i64 = get_digit(basek, *x);
            *c = get_carry(basek, *x, digit);
            *x = digit;
        });
    } else {
        let basek_lsh: usize = basek - lsh;
        x.iter_mut().zip(carry.iter_mut()).for_each(|(x, c)| {
            let digit: i64 = get_digit(basek_lsh, *x);
            *c = get_carry(basek_lsh, *x, digit);
            *x = digit << lsh;
        });
    }
}

#[inline(always)]
pub(crate) fn znx_normalize_first_step_ref(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
        assert_eq!(a.len(), carry.len());
        assert!(lsh < basek);
    }

    if lsh == 0 {
        izip!(x.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(x, a, c)| {
            let digit: i64 = get_digit(basek, *a);
            *c = get_carry(basek, *a, digit);
            *x = digit;
        });
    } else {
        let basek_lsh: usize = basek - lsh;
        izip!(x.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(x, a, c)| {
            let digit: i64 = get_digit(basek_lsh, *a);
            *c = get_carry(basek_lsh, *a, digit);
            *x = digit << lsh;
        });
    }
}

#[inline(always)]
pub(crate) fn znx_normalize_middle_step_carry_only_ref(basek: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
        assert!(lsh < basek);
    }
    if lsh == 0 {
        x.iter().zip(carry.iter_mut()).for_each(|(x, c)| {
            let digit: i64 = get_digit(basek, *x);
            let carry: i64 = get_carry(basek, *x, digit);
            let digit_plus_c: i64 = digit + *c;
            *c = carry + get_carry(basek, digit_plus_c, get_digit(basek, digit_plus_c));
        });
    } else {
        let basek_lsh: usize = basek - lsh;
        x.iter().zip(carry.iter_mut()).for_each(|(x, c)| {
            let digit: i64 = get_digit(basek_lsh, *x);
            let carry: i64 = get_carry(basek_lsh, *x, digit);
            let digit_plus_c: i64 = (digit << lsh) + *c;
            *c = carry + get_carry(basek, digit_plus_c, get_digit(basek, digit_plus_c));
        });
    }
}

#[inline(always)]
pub(crate) fn znx_normalize_middle_step_inplace_ref(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
        assert!(lsh < basek);
    }
    if lsh == 0 {
        x.iter_mut().zip(carry.iter_mut()).for_each(|(x, c)| {
            let digit: i64 = get_digit(basek, *x);
            let carry: i64 = get_carry(basek, *x, digit);
            let digit_plus_c: i64 = digit + *c;
            *x = get_digit(basek, digit_plus_c);
            *c = carry + get_carry(basek, digit_plus_c, *x);
        });
    } else {
        let basek_lsh: usize = basek - lsh;
        x.iter_mut().zip(carry.iter_mut()).for_each(|(x, c)| {
            let digit: i64 = get_digit(basek_lsh, *x);
            let carry: i64 = get_carry(basek_lsh, *x, digit);
            let digit_plus_c: i64 = (digit << lsh) + *c;
            *x = get_digit(basek, digit_plus_c);
            *c = carry + get_carry(basek, digit_plus_c, *x);
        });
    }
}

#[inline(always)]
pub(crate) fn znx_normalize_middle_step_ref(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
        assert_eq!(a.len(), carry.len());
        assert!(lsh < basek);
    }
    if lsh == 0 {
        izip!(x.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(x, a, c)| {
            let digit: i64 = get_digit(basek, *a);
            let carry: i64 = get_carry(basek, *a, digit);
            let digit_plus_c: i64 = digit + *c;
            *x = get_digit(basek, digit_plus_c);
            *c = carry + get_carry(basek, digit_plus_c, *x);
        });
    } else {
        let basek_lsh: usize = basek - lsh;
        izip!(x.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(x, a, c)| {
            let digit: i64 = get_digit(basek_lsh, *a);
            let carry: i64 = get_carry(basek_lsh, *a, digit);
            let digit_plus_c: i64 = (digit << lsh) + *c;
            *x = get_digit(basek, digit_plus_c);
            *c = carry + get_carry(basek, digit_plus_c, *x);
        });
    }
}

#[inline(always)]
pub(crate) fn znx_normalize_final_step_inplace_ref(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
        assert!(lsh < basek);
    }

    if lsh == 0 {
        x.iter_mut().zip(carry.iter_mut()).for_each(|(x, c)| {
            *x = get_digit(basek, get_digit(basek, *x) + *c);
        });
    } else {
        let basek_lsh: usize = basek - lsh;
        x.iter_mut().zip(carry.iter_mut()).for_each(|(x, c)| {
            *x = get_digit(basek, (get_digit(basek_lsh, *x) << lsh) + *c);
        });
    }
}

#[inline(always)]
pub(crate) fn znx_normalize_final_step_ref(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
        assert!(lsh < basek);
    }
    if lsh == 0 {
        izip!(x.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(x, a, c)| {
            *x = get_digit(basek, get_digit(basek, *a) + *c);
        });
    } else {
        let basek_lsh: usize = basek - lsh;
        izip!(x.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(x, a, c)| {
            *x = get_digit(basek, (get_digit(basek_lsh, *a) << lsh) + *c);
        });
    }
}
