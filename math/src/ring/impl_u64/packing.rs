use crate::modulus::barrett::Barrett;
use crate::modulus::{WordOps, ONCE};
use crate::poly::Poly;
use crate::ring::Ring;
use std::cmp::min;
use std::collections::HashSet;
use std::mem::transmute;

impl Ring<u64> {
    // Generates a vector storing {X^{2^0}, X^{2^1}, .., X^{2^log_n}}.
    pub fn gen_x_pow_2<const NTT: bool, const INV: bool>(&self, log_n: usize) -> Vec<Poly<u64>> {
        let mut x_pow: Vec<Poly<u64>> = Vec::<Poly<u64>>::with_capacity(log_n);

        (0..log_n).for_each(|i| {
            let mut idx: usize = 1 << i;

            if INV {
                idx = self.n() - idx;
            }

            x_pow.push(self.new_poly());

            if i == 0 {
                x_pow[i].0[idx] = self.modulus.montgomery.one();
                self.ntt_inplace::<false>(&mut x_pow[i]);
            } else {
                let (left, right) = x_pow.split_at_mut(i);
                self.a_mul_b_montgomery_into_c::<ONCE>(&left[i - 1], &left[i - 1], &mut right[0]);
            }
        });

        if INV {
            self.a_neg_into_a::<1, ONCE>(&mut x_pow[0]);
        }

        if !NTT {
            x_pow.iter_mut().for_each(|x| self.intt_inplace::<false>(x));
        }

        x_pow
    }

    pub fn pack<const ZEROGARBAGE: bool, const NTT: bool>(
        &self,
        polys: &mut [Option<Poly<u64>>],
        log_gap: usize,
    ) {
        let log_n: usize = self.log_n();
        let log_nth_root: usize = log_n + 1;
        let nth_root: usize = 1 << log_nth_root;
        let log_start: usize = log_n - log_gap;
        let mut log_end: usize = log_n;

        let mut indices: Vec<usize> = Vec::<usize>::new();

        // Retrives non-empty indexes
        polys.iter().enumerate().for_each(|(i, poly)| {
            if Some(poly) != None {
                indices.push(i);
            }
        });

        let gap: usize = max_gap(&indices);

        let set: HashSet<_> = indices.into_iter().collect();

        let max_pow2_gap_divisor: usize = 1 << gap.trailing_zeros();

        if !ZEROGARBAGE {
            if gap > 0 {
                log_end -= max_pow2_gap_divisor;
            }
        }

        let n_inv: Barrett<u64> = self
            .modulus
            .barrett
            .prepare(self.modulus.inv(1 << (log_end - log_start)));

        set.iter().for_each(|i| {
            if let Some(poly) = polys[*i].as_mut() {
                if !NTT {
                    self.ntt_inplace::<true>(poly);
                }
                self.a_mul_b_scalar_barrett_into_a::<ONCE>(&n_inv, poly);
            }
        });

        let x_pow2: Vec<Poly<u64>> = self.gen_x_pow_2::<true, false>(log_n);
        let mut tmpa: Poly<u64> = self.new_poly();
        let mut tmpb: Poly<u64> = self.new_poly();

        for i in log_start..log_end {
            let t: usize = 1 << (log_n - 1 - i);

            let (polys_lo, polys_hi) = polys.split_at_mut(t);

            for j in 0..t {
                if let Some(poly_hi) = polys_hi[j].as_mut() {
                    self.a_mul_b_montgomery_into_a::<ONCE>(&x_pow2[log_n - i - 1], poly_hi);

                    if let Some(poly_lo) = polys_lo[j].as_mut() {
                        self.a_sub_b_into_c::<1, ONCE>(poly_lo, poly_hi, &mut tmpa);
                        self.a_add_b_into_b::<ONCE>(poly_hi, poly_lo);
                    } else {
                        std::mem::swap(&mut polys_lo[j], &mut polys_hi[j]);
                    }
                }

                if let Some(poly_lo) = polys_lo[j].as_mut() {
                    let gal_el: usize = self.galois_element(1 << (i - 1), i == 0, log_nth_root);

                    if !polys_hi[j].is_none() {
                        self.automorphism::<true>(&tmpa, gal_el, 2 << self.log_n(), &mut tmpb);
                    } else {
                        self.automorphism::<true>(poly_lo, gal_el, nth_root, &mut tmpa);
                    }

                    self.a_add_b_into_b::<ONCE>(&tmpa, poly_lo);
                } else if let Some(poly_hi) = polys_hi[j].as_mut() {
                    let gal_el: usize = self.galois_element(1 << (i - 1), i == 0, log_nth_root);

                    self.automorphism::<true>(poly_hi, gal_el, nth_root, &mut tmpa);
                    self.a_sub_b_into_a::<1, ONCE>(&tmpa, poly_hi)
                }
            }
        }
    }
}

// Returns the largest gap.
fn max_gap(vec: &[usize]) -> usize {
    let mut gap: usize = usize::MAX;
    for i in 1..vec.len() {
        let (l, r) = (vec[i - 1], vec[i]);
        assert!(l > r, "invalid input vec: not sorted");
        gap = min(gap, r - l);
        if gap == 1 {
            break;
        }
    }
    gap
}
