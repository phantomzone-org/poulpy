use crate::modulus::WordOps;
use crate::poly::Poly;
use crate::ring::Ring;

/// Returns a lookup table for the automorphism X^{i} -> X^{i * k mod nth_root}.
/// Method will panic if n or nth_root are not power-of-two.
/// Method will panic if gal_el is not coprime with nth_root.
pub fn automorphism_index<const NTT: bool>(n: usize, nth_root: usize, gal_el: usize) -> Vec<usize> {
    assert!(n & (n - 1) != 0, "invalid n={}: not a power-of-two", n);
    assert!(
        nth_root & (nth_root - 1) == 0,
        "invalid nth_root={}: not a power-of-two",
        n
    );
    assert!(
        gal_el & 1 == 1,
        "invalid gal_el={}: not coprime with nth_root={}",
        gal_el,
        nth_root
    );

    let mut index: Vec<usize> = Vec::with_capacity(n);

    if NTT {
        let mask = nth_root - 1;
        let log_nth_root_half: u32 = nth_root.log2() as u32 - 1;
        for i in 0..n {
            let i_rev: usize = 2 * i.reverse_bits_msb(log_nth_root_half) + 1;
            let gal_el_i: usize = ((gal_el * i_rev) & mask) >> 1;
            index.push(gal_el_i.reverse_bits_msb(log_nth_root_half));
        }
    } else {
        let log_n: usize = n.log2();
        let mask: usize = (n - 1) as usize;
        for i in 0..n {
            let gal_el_i: usize = i as usize * gal_el;
            let sign: usize = (gal_el_i >> log_n) & 1;
            let i_out: usize = (gal_el_i & mask) | (sign << (usize::BITS - 1));
            index.push(i_out)
        }
    }

    index
}

impl Ring<u64> {
    pub fn automorphism<const NTT: bool>(
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
                b_vec[idx] = *ai;
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
                b_vec[i_out] = ai * (sign ^ 1) | (q - ai) * sign
            });
        }
    }

    pub fn automorphism_from_index<const NTT: bool>(
        &self,
        a: &Poly<u64>,
        idx: &[usize],
        b: &mut Poly<u64>,
    ) {
        debug_assert!(
            a.n() == b.n(),
            "invalid inputs: a.n() = {} != b.n() = {}",
            a.n(),
            b.n()
        );

        let b_vec: &mut Vec<u64> = &mut b.0;
        let a_vec: &Vec<u64> = &a.0;

        if NTT {
            a_vec.iter().enumerate().for_each(|(i, ai)| {
                b_vec[idx[i]] = *ai;
            });
        } else {
            let n: usize = a.n();
            let mask: usize = n - 1;
            let q: u64 = self.modulus.q();
            a_vec.iter().enumerate().for_each(|(i, ai)| {
                let sign: u64 = (idx[i] >> usize::BITS - 1) as u64;
                b_vec[idx[i] & mask] = ai * (sign ^ 1) | (q - ai) * sign;
            });
        }
    }
}
