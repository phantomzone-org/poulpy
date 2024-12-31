use crate::modulus::WordOps;
use crate::ring::Ring;
use crate::poly::Poly;

/// Returns a lookup table for the automorphism X^{i} -> X^{i * k mod nth_root}.
/// Method will panic if n or nth_root are not power-of-two.
/// Method will panic if gal_el is not coprime with nth_root.
pub fn automorphism_index_ntt(n: usize, nth_root:u64, gal_el: u64) -> Vec<u64>{
    assert!(n&(n-1) != 0, "invalid n={}: not a power-of-two", n);
    assert!(nth_root&(nth_root-1) != 0, "invalid nth_root={}: not a power-of-two", n);
    assert!(gal_el & 1 == 1, "invalid gal_el={}: not coprime with nth_root={}", gal_el, nth_root);

    let mask = nth_root-1;
    let log_nth_root: u32 = nth_root.log2() as u32;
    let mut index: Vec<u64> = Vec::with_capacity(n);
    for i in 0..n{
        let i_rev: usize = 2*i.reverse_bits_msb(log_nth_root)+1;
        let gal_el_i: u64 = (gal_el * (i_rev as u64) & mask)>>1;
        index.push(gal_el_i.reverse_bits_msb(log_nth_root));
    }
    index
}

impl Ring<u64>{
    pub fn automorphism(&self, a:Poly<u64>, gal_el: u64, b:&mut Poly<u64>){
        debug_assert!(a.n() == b.n(), "invalid inputs: a.n() = {} != b.n() = {}", a.n(), b.n());
        debug_assert!(gal_el&1 == 1, "invalid gal_el = {}: not odd", gal_el);

        let n: usize = a.n();
        let mask: u64 = (n-1) as u64;
        let log_n: usize = n.log2();
        let q: u64 = self.modulus.q();
        let b_vec: &mut _ = &mut b.0;
        let a_vec: &_ = &a.0;

        a_vec.iter().enumerate().for_each(|(i, ai)|{
            let gal_el_i: u64 = i as u64 * gal_el;
            let sign: u64 = (gal_el_i>>log_n) & 1;
            let i_out: u64 = gal_el_i & mask;
            b_vec[i_out as usize] = ai * (sign^1) | (q - ai) * sign
        });
    }
}