use crate::modulus::ONCE;
use crate::poly::Poly;
use crate::ring::Ring;

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
}
