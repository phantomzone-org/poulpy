use crate::dft::DFT;
use crate::modulus::barrett::Barrett;
use crate::modulus::montgomery::Montgomery;
use crate::modulus::prime::Prime;
use crate::modulus::ReduceOnce;
use crate::modulus::WordOps;
use crate::modulus::{BARRETT, NONE, ONCE};
use itertools::izip;

#[allow(dead_code)]
pub struct Table<O> {
    prime: Prime<O>,
    psi: O,
    psi_forward_rev: Vec<Barrett<u64>>,
    psi_backward_rev: Vec<Barrett<u64>>,
    q: O,
    two_q: O,
    four_q: O,
}

impl Table<u64> {
    pub fn new(prime: Prime<u64>, nth_root: u64) -> Table<u64> {
        assert!(
            nth_root & (nth_root - 1) == 0,
            "invalid argument: nth_root = {} is not a power of two",
            nth_root
        );

        let psi: u64 = prime.primitive_nth_root(nth_root);

        let psi_mont: Montgomery<u64> = prime.montgomery.prepare::<ONCE>(psi);
        let psi_inv_mont: Montgomery<u64> = prime.montgomery.pow(psi_mont, prime.phi - 1);

        let mut psi_forward_rev: Vec<Barrett<u64>> = vec![Barrett(0, 0); (nth_root >> 1) as usize];
        let mut psi_backward_rev: Vec<Barrett<u64>> = vec![Barrett(0, 0); (nth_root >> 1) as usize];

        psi_forward_rev[0] = prime.barrett.prepare(1);
        psi_backward_rev[0] = prime.barrett.prepare(1);

        let log_nth_root_half: u32 = (nth_root >> 1).log2() as _;

        let mut powers_forward: u64 = 1u64;
        let mut powers_backward: u64 = 1u64;

        for i in 1..(nth_root >> 1) as usize {
            let i_rev: usize = i.reverse_bits_msb(log_nth_root_half);

            prime
                .montgomery
                .mul_external_assign::<ONCE>(psi_mont, &mut powers_forward);
            prime
                .montgomery
                .mul_external_assign::<ONCE>(psi_inv_mont, &mut powers_backward);

            psi_forward_rev[i_rev] = prime.barrett.prepare(powers_forward);
            psi_backward_rev[i_rev] = prime.barrett.prepare(powers_backward);
        }

        let q: u64 = prime.q();

        Self {
            prime: prime,
            psi: psi,
            psi_forward_rev: psi_forward_rev,
            psi_backward_rev: psi_backward_rev,
            q: q,
            two_q: q << 1,
            four_q: q << 2,
        }
    }
}

impl DFT<u64> for Table<u64> {
    fn forward_inplace(&self, a: &mut [u64]) {
        self.forward_inplace::<false>(a)
    }

    fn forward_inplace_lazy(&self, a: &mut [u64]) {
        self.forward_inplace::<true>(a)
    }

    fn backward_inplace(&self, a: &mut [u64]) {
        self.backward_inplace::<false>(a)
    }

    fn backward_inplace_lazy(&self, a: &mut [u64]) {
        self.backward_inplace::<true>(a)
    }
}

impl Table<u64> {
    pub fn forward_inplace<const LAZY: bool>(&self, a: &mut [u64]) {
        self.forward_inplace_core::<LAZY, 0, 0>(a);
    }

    pub fn forward_inplace_core<const LAZY: bool, const SKIPSTART: u8, const SKIPEND: u8>(
        &self,
        a: &mut [u64],
    ) {
        let n: usize = a.len();
        assert!(
            n & n - 1 == 0,
            "invalid x.len()= {} must be a power of two",
            n
        );
        let log_n: u32 = usize::BITS - ((n as usize) - 1).leading_zeros();

        let start: u32 = SKIPSTART as u32;
        let end: u32 = log_n - (SKIPEND as u32);

        for layer in start..end {
            let (m, size) = (1 << layer, 1 << (log_n - layer - 1));
            let t: usize = 2 * size;
            if layer == log_n - 1 {
                if LAZY {
                    izip!(a.chunks_exact_mut(t), &self.psi_forward_rev[m..]).for_each(
                        |(a, psi)| {
                            let (a, b) = a.split_at_mut(size);
                            self.dit_inplace::<false>(&mut a[0], &mut b[0], *psi);
                            debug_assert!(
                                a[0] < self.two_q,
                                "forward_inplace_core::<LAZY=true> output {} > {} (2q-1)",
                                a[0],
                                self.two_q - 1
                            );
                            debug_assert!(
                                b[0] < self.two_q,
                                "forward_inplace_core::<LAZY=true> output {} > {} (2q-1)",
                                b[0],
                                self.two_q - 1
                            );
                        },
                    );
                } else {
                    izip!(a.chunks_exact_mut(t), &self.psi_forward_rev[m..]).for_each(
                        |(a, psi)| {
                            let (a, b) = a.split_at_mut(size);
                            self.dit_inplace::<true>(&mut a[0], &mut b[0], *psi);
                            self.prime.barrett.reduce_assign::<BARRETT>(&mut a[0]);
                            self.prime.barrett.reduce_assign::<BARRETT>(&mut b[0]);
                            debug_assert!(
                                a[0] < self.q,
                                "forward_inplace_core::<LAZY=false> output {} > {} (q-1)",
                                a[0],
                                self.q - 1
                            );
                            debug_assert!(
                                b[0] < self.q,
                                "forward_inplace_core::<LAZY=false> output {} > {} (q-1)",
                                b[0],
                                self.q - 1
                            );
                        },
                    );
                }
            } else if t >= 16 {
                izip!(a.chunks_exact_mut(t), &self.psi_forward_rev[m..]).for_each(|(a, psi)| {
                    let (a, b) = a.split_at_mut(size);
                    izip!(a.chunks_exact_mut(8), b.chunks_exact_mut(8)).for_each(|(a, b)| {
                        self.dit_inplace::<true>(&mut a[0], &mut b[0], *psi);
                        self.dit_inplace::<true>(&mut a[1], &mut b[1], *psi);
                        self.dit_inplace::<true>(&mut a[2], &mut b[2], *psi);
                        self.dit_inplace::<true>(&mut a[3], &mut b[3], *psi);
                        self.dit_inplace::<true>(&mut a[4], &mut b[4], *psi);
                        self.dit_inplace::<true>(&mut a[5], &mut b[5], *psi);
                        self.dit_inplace::<true>(&mut a[6], &mut b[6], *psi);
                        self.dit_inplace::<true>(&mut a[7], &mut b[7], *psi);
                    });
                });
            } else {
                izip!(a.chunks_exact_mut(t), &self.psi_forward_rev[m..]).for_each(|(a, psi)| {
                    let (a, b) = a.split_at_mut(size);
                    izip!(a, b).for_each(|(a, b)| self.dit_inplace::<true>(a, b, *psi));
                });
            }
        }
    }

    #[inline(always)]
    fn dit_inplace<const LAZY: bool>(&self, a: &mut u64, b: &mut u64, t: Barrett<u64>) {
        debug_assert!(*a < self.four_q, "a:{} q:{}", a, self.four_q);
        debug_assert!(*b < self.four_q, "b:{} q:{}", b, self.four_q);
        a.reduce_once_assign(self.two_q);
        let bt: u64 = self.prime.barrett.mul_external::<NONE>(t, *b);
        *b = *a + self.two_q - bt;
        *a += bt;
        if !LAZY {
            a.reduce_once_assign(self.two_q);
            b.reduce_once_assign(self.two_q);
        }
    }

    pub fn backward_inplace<const LAZY: bool>(&self, a: &mut [u64]) {
        self.backward_inplace_core::<LAZY, 0, 0>(a);
    }

    pub fn backward_inplace_core<const LAZY: bool, const SKIPSTART: u8, const SKIPEND: u8>(
        &self,
        a: &mut [u64],
    ) {
        let n: usize = a.len();
        assert!(
            n & n - 1 == 0,
            "invalid x.len()= {} must be a power of two",
            n
        );
        let log_n = usize::BITS - ((n as usize) - 1).leading_zeros();

        let start: u32 = SKIPEND as u32;
        let end: u32 = log_n - (SKIPSTART as u32);

        for layer in (start..end).rev() {
            let (m, size) = (1 << layer, 1 << (log_n - layer - 1));
            let t: usize = 2 * size;
            if layer == 0 {
                let n_inv: Barrett<u64> = self.prime.barrett.prepare(self.prime.inv(n as u64));
                let psi: Barrett<u64> = self.prime.barrett.prepare(
                    self.prime
                        .barrett
                        .mul_external::<ONCE>(n_inv, self.psi_backward_rev[1].0),
                );

                izip!(a.chunks_exact_mut(2 * size)).for_each(|a| {
                    let (a, b) = a.split_at_mut(size);
                    izip!(a.chunks_exact_mut(8), b.chunks_exact_mut(8)).for_each(|(a, b)| {
                        self.dif_last_inplace::<LAZY>(&mut a[0], &mut b[0], psi, n_inv);
                        self.dif_last_inplace::<LAZY>(&mut a[1], &mut b[1], psi, n_inv);
                        self.dif_last_inplace::<LAZY>(&mut a[2], &mut b[2], psi, n_inv);
                        self.dif_last_inplace::<LAZY>(&mut a[3], &mut b[3], psi, n_inv);
                        self.dif_last_inplace::<LAZY>(&mut a[4], &mut b[4], psi, n_inv);
                        self.dif_last_inplace::<LAZY>(&mut a[5], &mut b[5], psi, n_inv);
                        self.dif_last_inplace::<LAZY>(&mut a[6], &mut b[6], psi, n_inv);
                        self.dif_last_inplace::<LAZY>(&mut a[7], &mut b[7], psi, n_inv);
                    });
                });
            } else if t >= 16 {
                izip!(a.chunks_exact_mut(t), &self.psi_backward_rev[m..]).for_each(|(a, psi)| {
                    let (a, b) = a.split_at_mut(size);
                    izip!(a.chunks_exact_mut(8), b.chunks_exact_mut(8)).for_each(|(a, b)| {
                        self.dif_inplace::<true>(&mut a[0], &mut b[0], *psi);
                        self.dif_inplace::<true>(&mut a[1], &mut b[1], *psi);
                        self.dif_inplace::<true>(&mut a[2], &mut b[2], *psi);
                        self.dif_inplace::<true>(&mut a[3], &mut b[3], *psi);
                        self.dif_inplace::<true>(&mut a[4], &mut b[4], *psi);
                        self.dif_inplace::<true>(&mut a[5], &mut b[5], *psi);
                        self.dif_inplace::<true>(&mut a[6], &mut b[6], *psi);
                        self.dif_inplace::<true>(&mut a[7], &mut b[7], *psi);
                    });
                });
            } else {
                izip!(a.chunks_exact_mut(2 * size), &self.psi_backward_rev[m..]).for_each(
                    |(a, psi)| {
                        let (a, b) = a.split_at_mut(size);
                        izip!(a, b).for_each(|(a, b)| self.dif_inplace::<true>(a, b, *psi));
                    },
                );
            }
        }
    }

    #[inline(always)]
    fn dif_inplace<const LAZY: bool>(&self, a: &mut u64, b: &mut u64, t: Barrett<u64>) {
        debug_assert!(*a < self.two_q, "a:{} q:{}", a, self.two_q);
        debug_assert!(*b < self.two_q, "b:{} q:{}", b, self.two_q);
        let d: u64 = self
            .prime
            .barrett
            .mul_external::<NONE>(t, *a + self.two_q - *b);
        *a = *a + *b;
        a.reduce_once_assign(self.two_q);
        *b = d;
        if !LAZY {
            a.reduce_once_assign(self.q);
            b.reduce_once_assign(self.q);
        }
    }

    fn dif_last_inplace<const LAZY: bool>(
        &self,
        a: &mut u64,
        b: &mut u64,
        psi: Barrett<u64>,
        n_inv: Barrett<u64>,
    ) {
        debug_assert!(*a < self.two_q);
        debug_assert!(*b < self.two_q);
        if LAZY {
            let d: u64 = self
                .prime
                .barrett
                .mul_external::<NONE>(psi, *a + self.two_q - *b);
            *a = self.prime.barrett.mul_external::<NONE>(n_inv, *a + *b);
            *b = d;
        } else {
            let d: u64 = self
                .prime
                .barrett
                .mul_external::<ONCE>(psi, *a + self.two_q - *b);
            *a = self.prime.barrett.mul_external::<ONCE>(n_inv, *a + *b);
            *b = d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntt() {
        let q_base: u64 = 0x800000000004001;
        let q_power: usize = 1;
        let prime_instance: Prime<u64> = Prime::<u64>::new(q_base, q_power);
        let n: u64 = 32;
        let two_nth_root: u64 = n << 1;
        let ntt_table: Table<u64> = Table::<u64>::new(prime_instance, two_nth_root);
        let mut a: Vec<u64> = vec![0; n as usize];
        for i in 0..a.len() {
            a[i] = i as u64;
        }

        let b: Vec<u64> = a.clone();
        ntt_table.forward_inplace::<false>(&mut a);
        ntt_table.backward_inplace::<false>(&mut a);
        assert!(a == b);
    }
}
