use std::collections::HashMap;
use crate::poly::Poly;
use crate::ring::Ring;
use crate::modulus::{ONCE, WordOps};
use crate::modulus::barrett::Barrett;
use std::cmp::min;


impl Ring<u64>{

    // Generates a vector storing {X^{2^0}, X^{2^1}, .., X^{2^log_n}}.
    pub fn gen_x_pow_2<const NTT: bool, const INV: bool>(&self, log_n: usize) -> Vec<Poly<u64>>{
        let mut x_pow: Vec<Poly<u64>> = Vec::<Poly<u64>>::with_capacity(log_n);

        (0..log_n).for_each(|i|{
            let mut idx: usize = 1<<i;

            if INV{
                idx = self.n() - idx;
            }

            x_pow.push(self.new_poly());

            if i == 0{
                x_pow[i].0[idx] = self.modulus.montgomery.one();
                self.ntt_inplace::<false>(&mut x_pow[i]);
            }else{
                let (left, right) = x_pow.split_at_mut(i);
                self.a_mul_b_montgomery_into_c::<ONCE>(&left[i-1], &left[i-1], &mut right[0]);
            }
        });

        if INV{
            self.a_neg_into_a::<1, ONCE>(&mut x_pow[0]);
        }

        if !NTT{
            x_pow.iter_mut().for_each(|x| self.intt_inplace::<false>(x));
        }

        x_pow
    }

    pub fn pack<const ZEROGARBAGE: bool, const NTT: bool>(&self, polys: &mut HashMap<usize, Poly<u64>>, log_gap: usize) -> Poly<u64>{
        
        let log_n = self.log_n();
        let log_start = log_n - log_gap;
        let mut log_end = log_n;

        let mut keys: Vec<usize> = polys.keys().copied().collect();
        keys.sort();

        let mut gap = 0usize;

        if keys.len() > 1{
            gap = max_pow2_gap(&keys);
        }else{
            gap = 1<<log_n;
        }

        let log_gap: usize = gap.log2();

        if !ZEROGARBAGE{
            if gap > 0 {
                log_end -= log_gap;
            }
        }
        
        let n_inv: Barrett<u64> = self.modulus.barrett.prepare(self.modulus.inv(1<<(log_end - log_start)));

        for (_, poly) in polys.iter_mut() {
            if !NTT{
                self.ntt_inplace::<true>(poly);
            }

            self.a_mul_b_scalar_barrett_into_a::<ONCE>(&n_inv,  poly);
        }

        Poly::<u64>::default()
    }
}

// Returns the largest
fn max_pow2_gap(vec: &[usize]) -> usize{
    let mut gap: usize = usize::MAX;
    for i in 1..vec.len(){
        let (l, r) = (vec[i-1], vec[i]);
        assert!(l > r, "invalid input vec: not sorted");
        gap = min(gap, r-l);
        if gap == 1{
            break;
        }
    };
    1 << gap.trailing_zeros()
}