use crate::ring::Ring;
use crate::poly::Poly;
use crate::modulus::barrett::Barrett;
use crate::modulus::ONCE;

impl Ring<u64>{
    pub fn trace_inplace<const NTT:bool>(&self, log_start: usize, log_end: usize, a: &mut Poly<u64>){
        assert!(log_end <= self.log_n(), "invalid argument log_end: log_end={} > self.log_n()={}", log_end, self.log_n());
        assert!(log_end > log_start, "invalid argument log_start: log_start={} > log_end={}", log_start, log_end);

        let log_steps = log_end - log_start;

        if log_steps > 0 {
            let n_inv: Barrett<u64> = self.modulus.barrett.prepare(self.modulus.inv(1<<log_steps));
            self.a_mul_b_scalar_barrett_into_a::<ONCE>(&n_inv, a);

            if !NTT{
                self.ntt_inplace::<false>(a);
            }

            let mut tmp: Poly<u64> = self.new_poly();

            (log_start..log_end).for_each(|i|{

            });
        }

        

        
    }
}