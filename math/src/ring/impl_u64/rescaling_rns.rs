use crate::ring::RingRNS;
use crate::poly::{Poly, PolyRNS};
use crate::modulus::barrett::BarrettRNS;
use crate::modulus::ONCE;
extern crate test;

impl RingRNS<'_, u64>{

    /// Updates b to floor(a / q[b.level()]).
    /// Expects a and b to be in the NTT domain.
    pub fn div_floor_by_last_modulus_ntt(&self, a: &PolyRNS<u64>, buf: &mut PolyRNS<u64>, b: &mut PolyRNS<u64>){
        assert!(b.level() >= a.level()-1, "invalid input b: b.level()={} < a.level()-1={}", b.level(), a.level()-1);
        let level = self.level();
        self.0[level].intt::<false>(a.at(level), buf.at_mut(0));
        let rescaling_constants: BarrettRNS<u64> = self.rescaling_constant();
        let (buf_ntt_q_scaling, buf_ntt_qi_scaling) = buf.0.split_at_mut(1);
        for (i, r) in self.0[0..level].iter().enumerate(){
            r.ntt::<false>(&buf_ntt_q_scaling[0], &mut buf_ntt_qi_scaling[0]);
            r.sum_aqqmb_prod_c_scalar_barrett::<ONCE>(&buf_ntt_qi_scaling[0], a.at(i), &rescaling_constants.0[i], b.at_mut(i));
        }
    }

    /// Updates b to floor(b / q[b.level()]).
    /// Expects b to be in the NTT domain.
    pub fn div_floor_by_last_modulus_ntt_inplace(&self, buf: &mut PolyRNS<u64>, b: &mut PolyRNS<u64>){
        let level = self.level();
        self.0[level].intt::<true>(b.at(level), buf.at_mut(0));
        let rescaling_constants: BarrettRNS<u64> = self.rescaling_constant();
        let (buf_ntt_q_scaling, buf_ntt_qi_scaling) = buf.0.split_at_mut(1);
        for (i, r) in self.0[0..level].iter().enumerate(){
            r.ntt::<true>(&buf_ntt_q_scaling[0], &mut buf_ntt_qi_scaling[0]);
            r.sum_aqqmb_prod_c_scalar_barrett_inplace::<ONCE>(&buf_ntt_qi_scaling[0], &rescaling_constants.0[i], b.at_mut(i));
        }
    }

    /// Updates b to floor(a / q[b.level()]).
    pub fn div_floor_by_last_modulus(&self, a: &PolyRNS<u64>, b: &mut PolyRNS<u64>){
        assert!(b.level() >= a.level()-1, "invalid input b: b.level()={} < a.level()-1={}", b.level(), a.level()-1);
        let level = self.level();
        let rescaling_constants:crate::modulus::barrett::BarrettRNS<u64>  = self.rescaling_constant();
        for (i, r) in self.0[0..level].iter().enumerate(){
            r.sum_aqqmb_prod_c_scalar_barrett::<ONCE>(a.at(level), a.at(i), &rescaling_constants.0[i], b.at_mut(i));
        }
    }

    /// Updates a to floor(b / q[b.level()]).
    pub fn div_floor_by_last_modulus_inplace(&self, a: &mut PolyRNS<u64>){
        let level = self.level();
        let rescaling_constants: BarrettRNS<u64> = self.rescaling_constant();
        let (a_i, a_level) = a.split_at_mut(level);
        for (i, r) in self.0[0..level].iter().enumerate(){
            r.sum_aqqmb_prod_c_scalar_barrett_inplace::<ONCE>(&a_level[0], &rescaling_constants.0[i], &mut a_i[i]);
        }
    }

    pub fn div_floor_by_last_moduli(&self, nb_moduli:usize, a: &PolyRNS<u64>, b: &mut PolyRNS<u64>){
        if nb_moduli == 0{
            if a != b{
                b.copy(a);
            }
        }else{
            self.div_floor_by_last_modulus(a, b);
            (1..nb_moduli).for_each(|i|{self.at_level(self.level()-i).div_floor_by_last_modulus_inplace(b)});
        }
    }

    pub fn div_floor_by_last_moduli_inplace(&self, nb_moduli:usize, a: &mut PolyRNS<u64>){
        (0..nb_moduli).for_each(|i|{self.at_level(self.level()-i).div_floor_by_last_modulus_inplace(a)});
    }

    pub fn div_round_by_last_modulus_ntt(&self, a: &PolyRNS<u64>, buf: &mut PolyRNS<u64>, b: &mut PolyRNS<u64>){
        let level = self.level();
    }
}


#[cfg(test)]
mod tests {
    use num_bigint::BigInt;
    use crate::ring::Ring;
    use crate::ring::impl_u64::ring_rns::new_rings;
    use super::*;

    #[test]
    fn test_div_floor_by_last_modulus_ntt() {
        let n = 1<<10;
        let moduli: Vec<u64> = vec![0x1fffffffffc80001u64, 0x1fffffffffe00001u64];
        let rings: Vec<Ring<u64>> = new_rings(n, moduli);
        let ring_rns = RingRNS::new(&rings);

        let mut a: PolyRNS<u64> = ring_rns.new_polyrns();
        let mut b: PolyRNS<u64> = ring_rns.new_polyrns();
        let mut c: PolyRNS<u64> = ring_rns.at_level(ring_rns.level()-1).new_polyrns();

        // Allocates an rns poly with values [0..n]
        let mut coeffs_a: Vec<BigInt> = (0..n).map(|i|{BigInt::from(i)}).collect();
        ring_rns.from_bigint_inplace(&coeffs_a, 1, &mut a);

        // Scales by q_level both a and coeffs_a
        let scalar: u64 = ring_rns.0[ring_rns.level()].modulus.q;
        ring_rns.mul_scalar_inplace::<ONCE>(&scalar, &mut a);
        let scalar_big = BigInt::from(scalar);
        coeffs_a.iter_mut().for_each(|a|{*a *= &scalar_big});

        // Performs c = intt(ntt(a) / q_level)
        ring_rns.ntt_inplace::<false>(&mut a);
        ring_rns.div_floor_by_last_modulus_ntt(&a, &mut b, &mut c);
        ring_rns.at_level(c.level()).intt_inplace::<false>(&mut c);

        // Exports c to coeffs_c
        let mut coeffs_c = vec![BigInt::from(0);c.n()];
        ring_rns.at_level(c.level()).to_bigint_inplace(&c, 1, &mut coeffs_c);

        // Performs floor division on a
        coeffs_a.iter_mut().for_each(|a|{*a /= &scalar_big});

    	assert!(coeffs_a == coeffs_c);
    }
}