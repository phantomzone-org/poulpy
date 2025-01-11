use crate::ring::Ring;
use crate::poly::Poly;
use crate::modulus::barrett::Barrett;
use crate::modulus::ONCE;

impl Ring<u64>{
    pub fn trace_inplace<const NTT:bool>(&self, step_start: usize, a: &mut Poly<u64>){
        assert!(step_start <= self.log_n(), "invalid argument step_start: step_start={} > self.log_n()={}", step_start, self.log_n());

        let log_steps: usize = self.log_n() - step_start;
        let log_nth_root = self.log_n()+1;
        let nth_root: usize= 1<<log_nth_root;

        if log_steps > 0 {
            let n_inv: Barrett<u64> = self.modulus.barrett.prepare(self.modulus.inv(1<<log_steps));
            self.a_mul_b_scalar_barrett_into_a::<ONCE>(&n_inv, a);

            let mut tmp: Poly<u64> = self.new_poly();

            (step_start..self.log_n()).for_each(|i|{

                let gal_el: usize = self.galois_element((1 << i) >> 1, i == 0, log_nth_root);

                self.a_apply_automorphism_into_b::<NTT>(a, gal_el, nth_root, &mut tmp);
                self.a_add_b_into_b::<ONCE>(&tmp, a);
            });
        }
    }
}