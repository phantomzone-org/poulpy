use crate::ring::Ring;
use crate::ring::RingRNS;
use crate::poly::PolyRNS;
use crate::modulus::barrett::Barrett;
use crate::scalar::ScalarRNS;
use crate::modulus::ONCE;
extern crate test;

impl RingRNS<'_, u64>{

    /// Updates b to floor(a / q[b.level()]).
    pub fn div_floor_by_last_modulus<const NTT:bool>(&self, a: &PolyRNS<u64>, buf: &mut PolyRNS<u64>, b: &mut PolyRNS<u64>){
        debug_assert!(self.level() <= a.level(), "invalid input a: self.level()={} > a.level()={}", self.level(), a.level());
        debug_assert!(b.level() >= a.level()-1, "invalid input b: b.level()={} < a.level()-1={}", b.level(), a.level()-1);

        let level = self.level();
        let rescaling_constants: ScalarRNS<Barrett<u64>> = self.rescaling_constant();
        
        if NTT{
            let (buf_ntt_q_scaling, buf_ntt_qi_scaling) = buf.0.split_at_mut(1);
            self.0[level].intt::<false>(a.at(level), &mut buf_ntt_q_scaling[0]);
            for (i, r) in self.0[0..level].iter().enumerate(){
                r.ntt::<false>(&buf_ntt_q_scaling[0], &mut buf_ntt_qi_scaling[0]);
                r.sum_aqqmb_prod_c_scalar_barrett::<ONCE>(&buf_ntt_qi_scaling[0], a.at(i), &rescaling_constants.0[i], b.at_mut(i));
            }
        }else{
            for (i, r) in self.0[0..level].iter().enumerate(){
                r.sum_aqqmb_prod_c_scalar_barrett::<ONCE>(a.at(level), a.at(i), &rescaling_constants.0[i], b.at_mut(i));
            }
        }
    }

    /// Updates a to floor(a / q[b.level()]).
    /// Expects a to be in the NTT domain.
    pub fn div_floor_by_last_modulus_inplace<const NTT:bool>(&self, buf: &mut PolyRNS<u64>, a: &mut PolyRNS<u64>){
        debug_assert!(self.level() <= a.level(), "invalid input a: self.level()={} > a.level()={}", self.level(), a.level());

        let level = self.level();
        let rescaling_constants: ScalarRNS<Barrett<u64>> = self.rescaling_constant();

        if NTT{
            let (buf_ntt_q_scaling, buf_ntt_qi_scaling) = buf.0.split_at_mut(1);
            self.0[level].intt::<true>(a.at(level), &mut buf_ntt_q_scaling[0]);
            for (i, r) in self.0[0..level].iter().enumerate(){
                r.ntt::<true>(&buf_ntt_q_scaling[0], &mut buf_ntt_qi_scaling[0]);
                r.sum_aqqmb_prod_c_scalar_barrett_inplace::<ONCE>(&buf_ntt_qi_scaling[0], &rescaling_constants.0[i], a.at_mut(i));
            }
        }else{
            let (a_i, a_level) = buf.0.split_at_mut(level);
            for (i, r) in self.0[0..level].iter().enumerate(){
                r.sum_aqqmb_prod_c_scalar_barrett_inplace::<ONCE>(&a_level[0], &rescaling_constants.0[i], &mut a_i[i]);
            }
        }
    }

    /// Updates b to floor(a / prod_{level - nb_moduli}^{level} q[i])
    pub fn div_floor_by_last_moduli<const NTT:bool>(&self, nb_moduli:usize, a: &PolyRNS<u64>, buf: &mut PolyRNS<u64>, c: &mut PolyRNS<u64>){

        debug_assert!(self.level() <= a.level(), "invalid input a: self.level()={} > a.level()={}", self.level(), a.level());
        debug_assert!(c.level() >= a.level()-1, "invalid input b: b.level()={} < a.level()-1={}", c.level(), a.level()-1);
        debug_assert!(nb_moduli <= a.level(), "invalid input nb_moduli: nb_moduli={} > a.level()={}", nb_moduli, a.level());

        if nb_moduli == 0{
            if a != c{
                c.copy(a);
            }
        }else{
            if NTT{
                self.intt::<false>(a, buf);
                (0..nb_moduli).for_each(|i|{self.at_level(self.level()-i).div_floor_by_last_modulus_inplace::<false>(&mut PolyRNS::<u64>::default(), buf)});
                self.at_level(self.level()-nb_moduli).ntt::<false>(buf, c);
            }else{
                self.div_floor_by_last_modulus::<false>(a, buf, c);
                (1..nb_moduli).for_each(|i|{self.at_level(self.level()-i).div_floor_by_last_modulus_inplace::<false>(buf, c)});
            }
        }
    }

    /// Updates a to floor(a / prod_{level - nb_moduli}^{level} q[i])
    pub fn div_floor_by_last_moduli_inplace<const NTT:bool>(&self, nb_moduli:usize, buf: &mut PolyRNS<u64>, a: &mut PolyRNS<u64>){
        debug_assert!(self.level() <= a.level(), "invalid input a: self.level()={} > a.level()={}", self.level(), a.level());
        debug_assert!(nb_moduli <= a.level(), "invalid input nb_moduli: nb_moduli={} > a.level()={}", nb_moduli, a.level());
        if NTT{
            self.intt::<false>(a, buf);
            (0..nb_moduli).for_each(|i|{self.at_level(self.level()-i).div_floor_by_last_modulus_inplace::<false>(&mut PolyRNS::<u64>::default(), buf)});
            self.at_level(self.level()-nb_moduli).ntt::<false>(buf, a);
        }else{
            (0..nb_moduli).for_each(|i|{self.at_level(self.level()-i).div_floor_by_last_modulus_inplace::<false>(buf, a)});
        } 
    }

    /// Updates b to round(a / q[b.level()]).
    /// Expects b to be in the NTT domain.
    pub fn div_round_by_last_modulus<const NTT:bool>(&self, a: &PolyRNS<u64>, buf: &mut PolyRNS<u64>, b: &mut PolyRNS<u64>){
        debug_assert!(self.level() <= a.level(), "invalid input a: self.level()={} > a.level()={}", self.level(), a.level());
        debug_assert!(b.level() >= a.level()-1, "invalid input b: b.level()={} < a.level()-1={}", b.level(), a.level()-1);

        let level: usize = self.level();
        let r_last: &Ring<u64> = &self.0[level];
        let q_level_half: u64 = r_last.modulus.q>>1;
        let rescaling_constants: ScalarRNS<Barrett<u64>> = self.rescaling_constant();
        let (buf_ntt_q_scaling, buf_ntt_qi_scaling) = buf.0.split_at_mut(1);

        if NTT{
            r_last.intt::<false>(a.at(level), &mut buf_ntt_q_scaling[0]);
            r_last.add_scalar_inplace::<ONCE>(&q_level_half, &mut buf_ntt_q_scaling[0]);
            for (i, r) in self.0[0..level].iter().enumerate(){
                r_last.add_scalar::<ONCE>(&buf_ntt_q_scaling[0], &q_level_half, &mut buf_ntt_qi_scaling[0]);
                r.ntt_inplace::<false>(&mut buf_ntt_qi_scaling[0]);
                r.sum_aqqmb_prod_c_scalar_barrett::<ONCE>(&buf_ntt_qi_scaling[0], a.at(i), &rescaling_constants.0[i], b.at_mut(i));
            }
        }else{

        }
        
    }

    /// Updates a to round(a / q[b.level()]).
    /// Expects a to be in the NTT domain.
    pub fn div_round_by_last_modulus_inplace<const NTT:bool>(&self, buf: &mut PolyRNS<u64>, a: &mut PolyRNS<u64>){
        debug_assert!(self.level() <= a.level(), "invalid input a: self.level()={} > a.level()={}", self.level(), a.level());

        let level = self.level();
        let r_last: &Ring<u64> = &self.0[level];
        let q_level_half: u64 = r_last.modulus.q>>1;
        let rescaling_constants: ScalarRNS<Barrett<u64>> = self.rescaling_constant();
        let (buf_ntt_q_scaling, buf_ntt_qi_scaling) = buf.0.split_at_mut(1);

        if NTT{
            r_last.intt::<true>(a.at(level), &mut buf_ntt_q_scaling[0]);
            r_last.add_scalar_inplace::<ONCE>(&q_level_half, &mut buf_ntt_q_scaling[0]);
            for (i, r) in self.0[0..level].iter().enumerate(){
                r_last.add_scalar::<ONCE>(&buf_ntt_q_scaling[0], &q_level_half, &mut buf_ntt_qi_scaling[0]);
                r.ntt::<true>(&buf_ntt_q_scaling[0], &mut buf_ntt_qi_scaling[0]);
                r.sum_aqqmb_prod_c_scalar_barrett_inplace::<ONCE>(&buf_ntt_qi_scaling[0], &rescaling_constants.0[i], a.at_mut(i));
            }
        }
        
    }
}


