use crate::modulus::WordOps;

pub struct AutomorphismPermutation {
    pub gal_el: usize,
    pub permutation: Vec<usize>,
}

impl AutomorphismPermutation {
    /// Returns a lookup table for the automorphism X^{i} -> X^{i * k mod nth_root}.
    /// Method will panic if n or nth_root are not power-of-two.
    /// Method will panic if gal_el is not coprime with nth_root.
    pub fn new<const NTT: bool>(n: usize, gal_el: usize, nth_root: usize) -> Self {
        assert!(n & (n - 1) == 0, "invalid n={}: not a power-of-two", n);
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

        let mut permutation: Vec<usize> = Vec::with_capacity(n);

        if NTT {
            let mask = nth_root - 1;
            let log_nth_root_half: u32 = nth_root.log2() as u32 - 1;
            for i in 0..n {
                let i_rev: usize = 2 * i.reverse_bits_msb(log_nth_root_half) + 1;
                let gal_el_i: usize = ((gal_el * i_rev) & mask) >> 1;
                permutation.push(gal_el_i.reverse_bits_msb(log_nth_root_half));
            }
        } else {
            let log_n: usize = n.log2();
            let mask: usize = (n - 1) as usize;
            for i in 0..n {
                let gal_el_i: usize = i as usize * gal_el;
                let sign: usize = (gal_el_i >> log_n) & 1;
                let i_out: usize = (gal_el_i & mask) | (sign << (usize::BITS - 1));
                permutation.push(i_out)
            }
        }

        Self {
            gal_el: gal_el,
            permutation: permutation,
        }
    }
}
