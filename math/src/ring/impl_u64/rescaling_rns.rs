use crate::ring::RingRNS;
use crate::poly::PolyRNS;
use crate::modulus::barrett::Barrett;
use crate::modulus::ONCE;
extern crate test;

impl RingRNS<'_, u64>{

    /// Updates b to floor(b / q[b.level()]).
    /// Expects a and b to be in the NTT domain.
    pub fn div_floor_by_last_modulus_ntt(&self, a: &PolyRNS<u64>, buf: &mut PolyRNS<u64>, b: &mut PolyRNS<u64>){
        assert!(b.level() >= a.level()-1, "invalid input b: b.level()={} < a.level()-1={}", b.level(), a.level()-1);
        let level = self.level();
        self.0[level].intt::<true>(a.at(level), buf.at_mut(0));
        let rescaling_constants: Vec<Barrett<u64>> = self.rescaling_constant();
        let (buf_ntt_q_scaling, buf_ntt_qi_scaling) = buf.0.split_at_mut(1);
        for (i, r) in self.0[0..level].iter().enumerate(){
            r.ntt::<true>(&buf_ntt_q_scaling[0], &mut buf_ntt_qi_scaling[0]);
            r.sum_aqqmb_prod_c_scalar_barrett::<ONCE>(&buf_ntt_qi_scaling[0], a.at(i), &rescaling_constants[i], b.at_mut(i));
        }
    }

    /// Updates b to floor(b / q[b.level()]).
    pub fn div_floor_by_last_modulus(&self, a: &PolyRNS<u64>, b: &mut PolyRNS<u64>){
        assert!(b.level() >= a.level()-1, "invalid input b: b.level()={} < a.level()-1={}", b.level(), a.level()-1);
        let level = self.level();
        let rescaling_constants: Vec<Barrett<u64>> = self.rescaling_constant();
        for (i, r) in self.0[0..level].iter().enumerate(){
            r.sum_aqqmb_prod_c_scalar_barrett::<ONCE>(a.at(level), a.at(i), &rescaling_constants[i], b.at_mut(i));
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::ring::Ring;
    use crate::ring::impl_u64::ring_rns::new_rings;
    use super::*;

    #[test]
    fn test_div_floor_by_last_modulus_ntt() {
        let n = 1<<10;
        let moduli: Vec<u64> = vec![0x1fffffffffe00001u64, 0x1fffffffffc80001u64];
        let rings: Vec<Ring<u64>> = new_rings(n, moduli);
        let ring_rns = RingRNS::new(&rings);

        let a: PolyRNS<u64> = ring_rns.new_polyrns();
        let mut b: PolyRNS<u64> = ring_rns.new_polyrns();
        let mut c: PolyRNS<u64> = ring_rns.new_polyrns();

        ring_rns.div_floor_by_last_modulus_ntt(&a, &mut b, &mut c);

    	//assert!(m_precomp.mul_external::<ONCE>(y_mont, x) == (x as u128 * y as u128 % q as u128) as u64);
    }
}