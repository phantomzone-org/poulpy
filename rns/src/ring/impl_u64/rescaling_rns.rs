use crate::modulus::barrett::Barrett;
use crate::modulus::{BARRETT, NONE, ONCE};
use crate::poly::{Poly, PolyRNS};
use crate::ring::Ring;
use crate::ring::RingRNS;
use crate::scalar::ScalarRNS;
extern crate test;

impl RingRNS<u64> {
    /// Updates b to floor(a / q[b.level()]).
    /// buf is unused if <ROUND=false,NTT=false>
    pub fn div_by_last_modulus<const ROUND: bool, const NTT: bool>(
        &self,
        a: &PolyRNS<u64>,
        buf: &mut [Poly<u64>; 2],
        b: &mut PolyRNS<u64>,
    ) {
        debug_assert!(self.level() != 0, "invalid call: self.level()=0");
        debug_assert!(
            a.level() >= self.level(),
            "invalid input a:  a.level()={} < self.level()={}",
            a.level(),
            self.level()
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

        if ROUND {
            let q_level_half: u64 = r_last.modulus.q >> 1;

            let (buf_q_scaling, buf_qi_scaling) = buf.split_at_mut(1);

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
                r_last.a_add_b_scalar_into_c::<ONCE>(
                    a.at(self.level()),
                    &q_level_half,
                    &mut buf_q_scaling[0],
                );
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
        } else {
            if NTT {
                let (buf_ntt_q_scaling, buf_ntt_qi_scaling) = buf.split_at_mut(1);
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
        buf: &mut [Poly<u64>; 2],
        a: &mut PolyRNS<u64>,
    ) {
        debug_assert!(
            self.level() <= a.level(),
            "invalid input a: a.level()={} < self.level()={}",
            self.level(),
            a.level()
        );

        let level = self.level();
        let rescaling_constants: ScalarRNS<Barrett<u64>> = self.rescaling_constant();
        let r_last: &Ring<u64> = &self.0[level];

        if ROUND {
            let q_level_half: u64 = r_last.modulus.q >> 1;
            let (buf_q_scaling, buf_qi_scaling) = buf.split_at_mut(1);

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
        } else {
            if NTT {
                let (buf_ntt_q_scaling, buf_ntt_qi_scaling) = buf.split_at_mut(1);
                r_last.intt::<false>(a.at(level), &mut buf_ntt_q_scaling[0]);
                for (i, r) in self.0[0..level].iter().enumerate() {
                    r.ntt::<true>(&buf_ntt_q_scaling[0], &mut buf_ntt_qi_scaling[0]);
                    r.b_sub_a_mul_c_scalar_barrett_into_a::<2, ONCE>(
                        &buf_ntt_qi_scaling[0],
                        &rescaling_constants.0[i],
                        a.at_mut(i),
                    );
                }
            } else {
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
        nb_moduli_dropped: usize,
        a: &PolyRNS<u64>,
        buf0: &mut [Poly<u64>; 2],
        buf1: &mut PolyRNS<u64>,
        c: &mut PolyRNS<u64>,
    ) {
        debug_assert!(
            nb_moduli_dropped <= self.level(),
            "invalid input nb_moduli_dropped: nb_moduli_dropped={} > self.level()={}",
            nb_moduli_dropped,
            self.level()
        );
        debug_assert!(
            a.level() >= self.level(),
            "invalid input a: a.level()={} < self.level()={}",
            a.level(),
            self.level()
        );
        debug_assert!(
            buf1.level() >= self.level(),
            "invalid input buf: buf.level()={} < self.level()={}",
            buf1.level(),
            self.level()
        );
        debug_assert!(
            c.level() >= self.level() - nb_moduli_dropped,
            "invalid input c: c.level()={} < self.level()-nb_moduli_dropped={}",
            c.level(),
            self.level() - nb_moduli_dropped
        );

        if nb_moduli_dropped == 0 {
            if a != c {
                c.copy(a);
            }
        } else {
            if NTT {
                self.intt::<false>(a, buf1);
                (0..nb_moduli_dropped).for_each(|i| {
                    self.at_level(self.level() - i)
                        .div_by_last_modulus_inplace::<ROUND, false>(buf0, buf1)
                });
                self.at_level(self.level() - nb_moduli_dropped)
                    .ntt::<false>(buf1, c);
            } else {
                self.div_by_last_modulus::<ROUND, false>(a, buf0, buf1);

                (1..nb_moduli_dropped - 1).for_each(|i| {
                    self.at_level(self.level() - i)
                        .div_by_last_modulus_inplace::<ROUND, false>(buf0, buf1);
                });

                self.at_level(self.level() - nb_moduli_dropped + 1)
                    .div_by_last_modulus::<ROUND, false>(buf1, buf0, c);
            }
        }
    }

    /// Updates a to floor(a / prod_{level - nb_moduli_dropped}^{level} q[i])
    pub fn div_by_last_moduli_inplace<const ROUND: bool, const NTT: bool>(
        &self,
        nb_moduli_dropped: usize,
        buf0: &mut [Poly<u64>; 2],
        buf1: &mut PolyRNS<u64>,
        a: &mut PolyRNS<u64>,
    ) {
        debug_assert!(
            nb_moduli_dropped <= self.level(),
            "invalid input nb_moduli_dropped: nb_moduli_dropped={} > self.level()={}",
            nb_moduli_dropped,
            self.level()
        );
        debug_assert!(
            a.level() >= self.level(),
            "invalid input a: a.level()={} < self.level()={}",
            a.level(),
            self.level()
        );
        debug_assert!(
            buf1.level() >= self.level(),
            "invalid input buf: buf.level()={} < self.level()={}",
            buf1.level(),
            self.level()
        );
        if nb_moduli_dropped == 0 {
            return;
        }

        if NTT {
            self.intt::<false>(a, buf1);
            (0..nb_moduli_dropped).for_each(|i| {
                self.at_level(self.level() - i)
                    .div_by_last_modulus_inplace::<ROUND, false>(buf0, buf1)
            });
            self.at_level(self.level() - nb_moduli_dropped)
                .ntt::<false>(buf1, a);
        } else {
            (0..nb_moduli_dropped).for_each(|i| {
                self.at_level(self.level() - i)
                    .div_by_last_modulus_inplace::<ROUND, false>(buf0, a)
            });
        }
    }
}
