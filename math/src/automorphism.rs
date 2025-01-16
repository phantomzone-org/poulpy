use crate::modulus::WordOps;
use crate::ring::Ring;
use num::Unsigned;
use utils::map::Map;

pub struct AutoPermMap(Map<usize, AutoPerm>);

impl AutoPermMap {
    pub fn new() -> Self {
        Self {
            0: Map::<usize, AutoPerm>::new(),
        }
    }

    pub fn insert(&mut self, perm: AutoPerm) {
        self.0.insert(perm.gal_el, perm);
    }

    pub fn gen<O: Unsigned, const NTT: bool>(&mut self, ring: &Ring<O>, gen_1: usize, gen_2: bool) {
        self.insert(AutoPerm::new::<O, NTT>(ring, gen_1, gen_2))
    }

    pub fn get(&self, gal_el: &usize) -> Option<&AutoPerm> {
        self.0.get(gal_el)
    }
}

pub struct AutoPerm {
    pub gal_el: usize,
    pub permutation: Vec<usize>,
}

impl AutoPerm {
    /// Returns a lookup table for the automorphism X^{i} -> X^{i * k mod nth_root}.
    /// Method will panic if n or nth_root are not power-of-two.
    /// Method will panic if gal_el is not coprime with nth_root.
    pub fn new<O: Unsigned, const NTT: bool>(ring: &Ring<O>, gen_1: usize, gen_2: bool) -> Self {
        let n = ring.n();
        let cyclotomic_order = ring.cyclotomic_order();

        let gal_el = ring.galois_element(gen_1, gen_2);

        let mut permutation: Vec<usize> = Vec::with_capacity(n);

        if NTT {
            let mask = cyclotomic_order - 1;
            let log_cyclotomic_order_half: u32 = cyclotomic_order.log2() as u32 - 1;
            for i in 0..n {
                let i_rev: usize = 2 * i.reverse_bits_msb(log_cyclotomic_order_half) + 1;
                let gal_el_i: usize = ((gal_el * i_rev) & mask) >> 1;
                permutation.push(gal_el_i.reverse_bits_msb(log_cyclotomic_order_half));
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
