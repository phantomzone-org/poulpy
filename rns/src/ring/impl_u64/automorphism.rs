use crate::automorphism::AutoPerm;
use crate::modulus::{ScalarOperations, ONCE};
use crate::modulus::{WordOps, REDUCEMOD};
use crate::poly::Poly;
use crate::ring::Ring;

impl Ring<u64> {
    // b <- auto(a)
    pub fn a_apply_automorphism_native_into_b<const NTT: bool>(
        &self,
        a: &Poly<u64>,
        gal_el: usize,
        nth_root: usize,
        b: &mut Poly<u64>,
    ) {
        self.apply_automorphism_native_core::<0, ONCE, NTT>(a, gal_el, nth_root, b)
    }

    // b <- REDUCEMOD(b + auto(a))
    pub fn a_apply_automorphism_native_add_b_into_b<const REDUCE: REDUCEMOD, const NTT: bool>(
        &self,
        a: &Poly<u64>,
        gal_el: usize,
        nth_root: usize,
        b: &mut Poly<u64>,
    ) {
        self.apply_automorphism_native_core::<1, REDUCE, NTT>(a, gal_el, nth_root, b)
    }

    // b <- REDUCEMOD(b - auto(a))
    pub fn a_apply_automorphism_native_sub_b_into_b<const REDUCE: REDUCEMOD, const NTT: bool>(
        &self,
        a: &Poly<u64>,
        gal_el: usize,
        nth_root: usize,
        b: &mut Poly<u64>,
    ) {
        self.apply_automorphism_native_core::<2, REDUCE, NTT>(a, gal_el, nth_root, b)
    }

    fn apply_automorphism_native_core<const MOD: u8, const REDUCE: REDUCEMOD, const NTT: bool>(
        &self,
        a: &Poly<u64>,
        gal_el: usize,
        nth_root: usize,
        b: &mut Poly<u64>,
    ) {
        debug_assert!(
            a.n() == b.n(),
            "invalid inputs: a.n() = {} != b.n() = {}",
            a.n(),
            b.n()
        );

        assert!(
            gal_el & 1 == 1,
            "invalid gal_el={}: not coprime with nth_root={}",
            gal_el,
            nth_root
        );

        assert!(
            nth_root & (nth_root - 1) == 0,
            "invalid nth_root={}: not a power-of-two",
            nth_root
        );

        let b_vec: &mut Vec<u64> = &mut b.0;
        let a_vec: &Vec<u64> = &a.0;

        if NTT {
            let mask: usize = nth_root - 1;
            let log_nth_root_half: u32 = nth_root.log2() as u32 - 1;
            a_vec.iter().enumerate().for_each(|(i, ai)| {
                let i_rev: usize = 2 * i.reverse_bits_msb(log_nth_root_half) + 1;
                let gal_el_i: usize = (((gal_el * i_rev) & mask) - 1) >> 1;
                let idx: usize = gal_el_i.reverse_bits_msb(log_nth_root_half);
                match MOD {
                    0 => b_vec[idx] = *ai,
                    1 => self
                        .modulus
                        .sa_add_sb_into_sb::<REDUCE>(ai, &mut b_vec[idx]),
                    2 => self
                        .modulus
                        .sa_sub_sb_into_sa::<1, REDUCE>(ai, &mut b_vec[idx]),
                    _ => {
                        panic!("invalid const MOD should be 0, 1, or 2 but is {}", MOD)
                    }
                }
            });
        } else {
            let n: usize = a.n();
            let mask: usize = n - 1;
            let log_n: usize = n.log2();
            let q: u64 = self.modulus.q();
            a_vec.iter().enumerate().for_each(|(i, ai)| {
                let gal_el_i: usize = i * gal_el;
                let sign: u64 = ((gal_el_i >> log_n) & 1) as u64;
                let i_out: usize = gal_el_i & mask;
                let v: u64 = ai * (sign ^ 1) | (q - ai) * sign;
                match MOD {
                    0 => b_vec[i_out] = v,
                    1 => self
                        .modulus
                        .sa_add_sb_into_sb::<REDUCE>(&v, &mut b_vec[i_out]),
                    2 => self
                        .modulus
                        .sa_sub_sb_into_sa::<1, REDUCE>(&v, &mut b_vec[i_out]),
                    _ => {
                        panic!("invalid const MOD should be 0, 1, or 2 but is {}", MOD)
                    }
                }
            });
        }
    }

    // b <- auto(a)
    pub fn a_apply_automorphism_from_perm_into_b<const NTT: bool>(
        &self,
        a: &Poly<u64>,
        auto_perm: &AutoPerm,
        b: &mut Poly<u64>,
    ) {
        self.automorphism_from_perm_core::<0, ONCE, NTT>(a, auto_perm, b)
    }

    // b <- REDUCEMOD(b + auto(a))
    pub fn a_apply_automorphism_from_perm_add_b_into_b<const REDUCE: REDUCEMOD, const NTT: bool>(
        &self,
        a: &Poly<u64>,
        auto_perm: &AutoPerm,
        b: &mut Poly<u64>,
    ) {
        self.automorphism_from_perm_core::<1, REDUCE, NTT>(a, auto_perm, b)
    }

    // b <- REDUCEMOD(b - auto(a))
    pub fn a_apply_automorphism_from_perm_sub_b_into_b<const REDUCE: REDUCEMOD, const NTT: bool>(
        &self,
        a: &Poly<u64>,
        auto_perm: &AutoPerm,
        b: &mut Poly<u64>,
    ) {
        self.automorphism_from_perm_core::<2, REDUCE, NTT>(a, auto_perm, b)
    }

    // b <- auto(a) if OVERWRITE else b <- REDUCEMOD(b + auto(a))
    fn automorphism_from_perm_core<const MOD: u8, const REDUCE: REDUCEMOD, const NTT: bool>(
        &self,
        a: &Poly<u64>,
        auto_perm: &AutoPerm,
        b: &mut Poly<u64>,
    ) {
        debug_assert!(
            a.n() == b.n(),
            "invalid inputs: a.n() = {} != b.n() = {}",
            a.n(),
            b.n()
        );

        assert!(
            NTT == auto_perm.ntt,
            "missmatch between AutoPerm NTT flag={} and method NTT flag={}",
            auto_perm.ntt,
            NTT
        );

        let b_vec: &mut Vec<u64> = &mut b.0;
        let a_vec: &Vec<u64> = &a.0;

        let idx: &Vec<usize> = &auto_perm.permutation;

        if NTT {
            a_vec.iter().enumerate().for_each(|(i, ai)| match MOD {
                0 => b_vec[idx[i]] = *ai,
                1 => self
                    .modulus
                    .sa_add_sb_into_sb::<REDUCE>(ai, &mut b_vec[idx[i]]),
                2 => self
                    .modulus
                    .sa_sub_sb_into_sa::<1, REDUCE>(ai, &mut b_vec[idx[i]]),
                _ => {
                    panic!("invalid const MOD should be 0, 1, or 2 but is {}", MOD)
                }
            });
        } else {
            let n: usize = a.n();
            let mask: usize = n - 1;
            let q: u64 = self.modulus.q();
            a_vec.iter().enumerate().for_each(|(i, ai)| {
                let sign: u64 = (idx[i] >> usize::BITS - 1) as u64;
                let v: u64 = ai * (sign ^ 1) | (q - ai) * sign;
                match MOD {
                    0 => b_vec[idx[i] & mask] = v,
                    1 => self
                        .modulus
                        .sa_add_sb_into_sb::<REDUCE>(&v, &mut b_vec[idx[i] & mask]),
                    2 => self
                        .modulus
                        .sa_sub_sb_into_sa::<1, REDUCE>(&v, &mut b_vec[idx[i] & mask]),
                    _ => {
                        panic!("invalid const MOD should be 0, 1, or 2 but is {}", MOD)
                    }
                }
            });
        }
    }
}
