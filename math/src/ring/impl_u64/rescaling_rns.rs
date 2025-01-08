use crate::modulus::barrett::Barrett;
use crate::modulus::{NONE, ONCE, BARRETT};
use crate::poly::PolyRNS;
use crate::ring::Ring;
use crate::ring::RingRNS;
use crate::scalar::ScalarRNS;
extern crate test;

impl RingRNS<u64> {
    /// Updates b to floor(a / q[b.level()]).
    pub fn div_by_last_modulus<const ROUND: bool, const NTT: bool>(
        &self,
        a: &PolyRNS<u64>,
        buf: &mut PolyRNS<u64>,
        b: &mut PolyRNS<u64>,
    ) {
        debug_assert!(self.level() != 0, "invalid call: self.level()=0");
        debug_assert!(
            self.level() <= a.level(),
            "invalid input a: self.level()={} > a.level()={}",
            self.level(),
            a.level()
        );
        debug_assert!(
            b.level() >= self.level() - 1,
            "invalid input b: b.level()={} < self.level()-1={}",
            b.level(),
            self.level() - 1
        );

        let level = self.level();
        let rescaling_constants: ScalarRNS<Barrett<u64>> = self.rescaling_constant();
        let r_last: &Ring<u64> = &self.0[level];
        
        if ROUND{

            let q_level_half: u64 = r_last.modulus.q >> 1;

            let (buf_q_scaling, buf_qi_scaling) = buf.0.split_at_mut(1);

            if NTT {
                r_last.intt::<false>(a.at(level), &mut buf_q_scaling[0]);
                r_last.a_add_b_scalar_into_a::<ONCE>(&q_level_half, &mut buf_q_scaling[0]);
                for (i, r) in self.0[0..level].iter().enumerate() {
                    r_last.a_add_b_scalar_into_c::<NONE>(
                        &buf_q_scaling[0],
                        &(r.modulus.q - r_last.modulus.barrett.reduce::<BARRETT>(&q_level_half)),
                        &mut buf_qi_scaling[0],
                    );
                    r.ntt_inplace::<true>(&mut buf_qi_scaling[0]);
                    r.a_sub_b_mul_c_scalar_barrett_into_d::<2, ONCE>(
                        &buf_qi_scaling[0],
                        a.at(i),
                        &rescaling_constants.0[i],
                        b.at_mut(i),
                    );
                }
            } else {
                r_last.a_add_b_scalar_into_c::<ONCE>(a.at(self.level()), &q_level_half, &mut buf_q_scaling[0]);
                for (i, r) in self.0[0..level].iter().enumerate() {
                    r_last.a_add_b_scalar_into_c::<NONE>(
                        &buf_q_scaling[0],
                        &(r.modulus.q - r_last.modulus.barrett.reduce::<BARRETT>(&q_level_half)),
                        &mut buf_qi_scaling[0],
                    );
                    r.a_sub_b_mul_c_scalar_barrett_into_d::<2, ONCE>(
                        &buf_qi_scaling[0],
                        a.at(i),
                        &rescaling_constants.0[i],
                        b.at_mut(i),
                    );
                }
            }
        }else{
            if NTT {
                let (buf_ntt_q_scaling, buf_ntt_qi_scaling) = buf.0.split_at_mut(1);
                self.0[level].intt::<false>(a.at(level), &mut buf_ntt_q_scaling[0]);
                for (i, r) in self.0[0..level].iter().enumerate() {
                    r.ntt::<true>(&buf_ntt_q_scaling[0], &mut buf_ntt_qi_scaling[0]);
                    r.a_sub_b_mul_c_scalar_barrett_into_d::<2, ONCE>(
                        &buf_ntt_qi_scaling[0],
                        a.at(i),
                        &rescaling_constants.0[i],
                        b.at_mut(i),
                    );
                }
            } else {
                for (i, r) in self.0[0..level].iter().enumerate() {
                    r.a_sub_b_mul_c_scalar_barrett_into_d::<2, ONCE>(
                        a.at(level),
                        a.at(i),
                        &rescaling_constants.0[i],
                        b.at_mut(i),
                    );
                }
            }
        }
    }

    /// Updates a to floor(a / q[b.level()]).
    /// Expects a to be in the NTT domain.
    pub fn div_by_last_modulus_inplace<const ROUND: bool, const NTT: bool>(
        &self,
        buf: &mut PolyRNS<u64>,
        a: &mut PolyRNS<u64>,
    ) {
        debug_assert!(
            self.level() <= a.level(),
            "invalid input a: self.level()={} > a.level()={}",
            self.level(),
            a.level()
        );

        let level = self.level();
        let rescaling_constants: ScalarRNS<Barrett<u64>> = self.rescaling_constant();
        let r_last: &Ring<u64> = &self.0[level];

        if ROUND{

            let q_level_half: u64 = r_last.modulus.q >> 1;
            let (buf_q_scaling, buf_qi_scaling) = buf.0.split_at_mut(1);

            if NTT {
                r_last.intt::<false>(a.at(level), &mut buf_q_scaling[0]);
                r_last.a_add_b_scalar_into_a::<ONCE>(&q_level_half, &mut buf_q_scaling[0]);
                for (i, r) in self.0[0..level].iter().enumerate() {
                    r_last.a_add_b_scalar_into_c::<NONE>(
                        &buf_q_scaling[0],
                        &(r.modulus.q - r_last.modulus.barrett.reduce::<BARRETT>(&q_level_half)),
                        &mut buf_qi_scaling[0],
                    );
                    r.ntt_inplace::<false>(&mut buf_qi_scaling[0]);
                    r.b_sub_a_mul_c_scalar_barrett_into_a::<2, ONCE>(
                        &buf_qi_scaling[0],
                        &rescaling_constants.0[i],
                        a.at_mut(i),
                    );
                }
            } else {
                let (a_qi, a_q_last) = a.0.split_at_mut(self.level());
                r_last.a_add_b_scalar_into_a::<ONCE>(&q_level_half, &mut a_q_last[0]);
                for (i, r) in self.0[0..level].iter().enumerate() {
                    r.b_sub_a_add_c_scalar_mul_d_scalar_barrett_into_a::<1, ONCE>(
                        &a_q_last[0],
                        &(r.modulus.q - r_last.modulus.barrett.reduce::<BARRETT>(&q_level_half)),
                        &rescaling_constants.0[i],
                        &mut a_qi[i],
                    );
                }
            }
        }else{

            if NTT {
                let (buf_ntt_q_scaling, buf_ntt_qi_scaling) = buf.0.split_at_mut(1);
                r_last.intt::<false>(a.at(level), &mut buf_ntt_q_scaling[0]);
                for (i, r) in self.0[0..level].iter().enumerate() {
                    r.ntt::<true>(&buf_ntt_q_scaling[0], &mut buf_ntt_qi_scaling[0]);
                    r.b_sub_a_mul_c_scalar_barrett_into_a::<2, ONCE>(
                        &buf_ntt_qi_scaling[0],
                        &rescaling_constants.0[i],
                        a.at_mut(i),
                    );
                }
            }else{
                let (a_i, a_level) = a.0.split_at_mut(level);
                for (i, r) in self.0[0..level].iter().enumerate() {
                    r.b_sub_a_mul_c_scalar_barrett_into_a::<2, ONCE>(
                        &a_level[0],
                        &rescaling_constants.0[i],
                        &mut a_i[i],
                    );
                }
            }
        }
        
    }

    /// Updates b to floor(a / prod_{level - nb_moduli}^{level} q[i])
    pub fn div_by_last_moduli<const ROUND: bool, const NTT: bool>(
        &self,
        nb_moduli: usize,
        a: &PolyRNS<u64>,
        buf: &mut PolyRNS<u64>,
        c: &mut PolyRNS<u64>,
    ) {
        debug_assert!(
            nb_moduli <= self.level(),
            "invalid input nb_moduli: nb_moduli={} > a.level()={}",
            nb_moduli,
            a.level()
        );
        debug_assert!(
            a.level() <= self.level(),
            "invalid input a: self.level()={} > a.level()={}",
            self.level(),
            a.level()
        );
        debug_assert!(
            buf.level() >= self.level() - 1,
            "invalid input buf: buf.level()={} < a.level()-1={}",
            buf.level(),
            a.level() - 1
        );
        debug_assert!(
            c.level() >= self.level() - nb_moduli,
            "invalid input c: c.level()={} < c.level()-nb_moduli={}",
            c.level(),
            a.level() - nb_moduli
        );
    
        if nb_moduli == 0 {
            if a != c {
                c.copy(a);
            }
        } else {

            if NTT{
                self.intt::<false>(a, buf);
                (0..nb_moduli).for_each(|i| {
                    self.at_level(self.level() - i)
                        .div_by_last_modulus_inplace::<ROUND, false>(
                            &mut PolyRNS::<u64>::default(),
                            buf,
                        )
                });
                self.at_level(self.level() - nb_moduli).ntt::<false>(buf, c);
            }else{
                
                println!("{} {:?}", self.level(), buf.level());
                self.div_by_last_modulus::<ROUND, false>(a, buf, c);

                (1..nb_moduli-1).for_each(|i| {
                    println!("{} {:?}", self.level() - i, buf.level());
                    self.at_level(self.level() - i)
                        .div_by_last_modulus_inplace::<ROUND, false>(buf, c);
                });
                
                self.at_level(self.level()-nb_moduli+1).div_by_last_modulus_inplace::<ROUND, false>(buf, c);
            }
        }
    }

    /// Updates a to floor(a / prod_{level - nb_moduli}^{level} q[i])
    pub fn div_by_last_moduli_inplace<const ROUND:bool, const NTT: bool>(
        &self,
        nb_moduli: usize,
        buf: &mut PolyRNS<u64>,
        a: &mut PolyRNS<u64>,
    ) {
        debug_assert!(
            self.level() <= a.level(),
            "invalid input a: self.level()={} > a.level()={}",
            self.level(),
            a.level()
        );
        debug_assert!(
            nb_moduli <= a.level(),
            "invalid input nb_moduli: nb_moduli={} > a.level()={}",
            nb_moduli,
            a.level()
        );
        if nb_moduli == 0{
            return
        }

        if NTT {
            self.intt::<false>(a, buf);
            (0..nb_moduli).for_each(|i| {
                self.at_level(self.level() - i)
                    .div_by_last_modulus_inplace::<ROUND, false>(&mut PolyRNS::<u64>::default(), buf)
            });
            self.at_level(self.level() - nb_moduli).ntt::<false>(buf, a);
        } else {
            (0..nb_moduli).for_each(|i| {
                self.at_level(self.level() - i)
                    .div_by_last_modulus_inplace::<ROUND, false>(buf, a)
            });
        }
    }
}
