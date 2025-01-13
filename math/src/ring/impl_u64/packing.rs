use crate::modulus::barrett::Barrett;
use crate::modulus::montgomery::Montgomery;
use crate::modulus::ONCE;
use crate::poly::Poly;
use crate::ring::Ring;
use std::cmp::min;
use std::rc::Rc;

impl Ring<u64> {
    pub fn pack<const ZEROGARBAGE: bool, const NTT: bool>(
        &self,
        polys: &mut Vec<Option<Poly<u64>>>,
        log_gap: usize,
    ) {
        let log_n: usize = self.log_n();
        let log_nth_root: usize = log_n + 1;
        let nth_root: usize = 1 << log_nth_root;
        let log_start: usize = log_n - log_gap;
        let mut log_end: usize = log_n;

        let mut indices: Vec<usize> = Vec::<usize>::new();

        // Retrives non-empty indexes
        polys.iter().enumerate().for_each(|(i, poly)| {
            if !poly.is_none() {
                indices.push(i);
            }
        });

        let gap: usize = max_gap(&indices);

        if !ZEROGARBAGE {
            if gap > 0 {
                log_end -= gap.trailing_zeros() as usize;
            }
        }

        assert!(
            log_start < log_end,
            "invalid input polys: gap between non None value is smaller than 2^log_gap"
        );

        let n_inv: Barrett<u64> = self
            .modulus
            .barrett
            .prepare(self.modulus.inv(1 << (log_end - log_start)));

        indices.iter().for_each(|i| {
            if let Some(poly) = polys[*i].as_mut() {
                if !NTT {
                    self.ntt_inplace::<true>(poly);
                }
                self.a_mul_b_scalar_barrett_into_a::<ONCE>(&n_inv, poly);
            }
        });

        let x_pow2: Vec<Poly<Montgomery<u64>>> = self.gen_x_pow_2::<true, false>(log_n);
        let mut tmpa: Poly<u64> = self.new_poly();
        let mut tmpb: Poly<u64> = self.new_poly();

        for i in log_start..log_end {
            let t: usize = 1 << (log_n - 1 - i);

            let (polys_lo, polys_hi) = polys.split_at_mut(t);

            for j in 0..t {
                if let Some(poly_hi) = polys_hi[j].as_mut() {
                    self.a_mul_b_montgomery_into_a::<ONCE>(&x_pow2[log_n - i - 1], poly_hi);

                    if let Some(poly_lo) = polys_lo[j].as_mut() {
                        self.a_sub_b_into_c::<1, ONCE>(poly_lo, poly_hi, &mut tmpa);
                        self.a_add_b_into_b::<ONCE>(poly_hi, poly_lo);
                    }
                }

                if let Some(poly_lo) = polys_lo[j].as_mut() {
                    let gal_el: usize = self.galois_element((1 << i) >> 1, i == 0, log_nth_root);

                    if !polys_hi[j].is_none() {
                        self.a_apply_automorphism_add_b_into_b::<ONCE, true>(
                            &tmpa,
                            gal_el,
                            2 << self.log_n(),
                            poly_lo,
                        );
                    } else {
                        self.a_apply_automorphism_into_b::<true>(
                            poly_lo, gal_el, nth_root, &mut tmpa,
                        );
                        self.a_add_b_into_b::<ONCE>(&tmpa, poly_lo);
                    }
                } else if let Some(poly_hi) = polys_hi[j].as_mut() {
                    let gal_el: usize = self.galois_element((1 << i) >> 1, i == 0, log_nth_root);
                    self.a_apply_automorphism_into_b::<true>(poly_hi, gal_el, nth_root, &mut tmpa);
                    self.a_sub_b_into_a::<1, ONCE>(&tmpa, poly_hi);
                    std::mem::swap(&mut polys_lo[j], &mut polys_hi[j]);
                }
            }

            polys.truncate(t);
        }

        if !NTT {
            if let Some(poly) = polys[0].as_mut() {
                self.intt_inplace::<false>(poly);
            }
        }
    }
}

// Returns the largest gap between two values in an ordered array of distinct values.
// Panics if the array is not ordered or values are not distincts.
fn max_gap(vec: &[usize]) -> usize {
    let mut gap: usize = usize::MAX;
    for i in 1..vec.len() {
        let (l, r) = (vec[i - 1], vec[i]);
        assert!(
            r > l,
            "invalid input vec: not sorted or collision between indices"
        );
        gap = min(gap, r - l);
        if gap == 1 {
            break;
        }
    }
    gap
}

pub struct StreamRepacker {
    accumulators: Vec<Accumulator>,
    tmp_a: Poly<u64>,
    tmp_b: Poly<u64>,
    x_pow_2: Vec<Poly<Montgomery<u64>>>,
    n_inv: Barrett<u64>,
    pub results: Vec<Poly<u64>>,
    counter: usize,
}

pub struct Accumulator {
    buf: Poly<u64>,
    value: bool,
    control: bool,
}

impl Accumulator {
    pub fn new(r: &Ring<u64>) -> Self {
        Self {
            buf: r.new_poly(),
            value: false,
            control: false,
        }
    }
}

impl StreamRepacker {
    pub fn new(r: &Ring<u64>) -> Self {
        let mut accumulators: Vec<Accumulator> = Vec::<Accumulator>::new();

        (0..r.log_n()).for_each(|_| accumulators.push(Accumulator::new(r)));

        Self {
            accumulators: accumulators,
            tmp_a: r.new_poly(),
            tmp_b: r.new_poly(),
            x_pow_2: r.gen_x_pow_2::<true, false>(r.log_n()),
            n_inv: r.modulus.barrett.prepare(r.modulus.inv(r.n() as u64)),
            results: Vec::<Poly<u64>>::new(),
            counter: 0,
        }
    }

    pub fn reset(&mut self) {
        for i in 0..self.accumulators.len() {
            self.accumulators[i].value = false;
            self.accumulators[i].control = false;
        }
        self.counter = 0;
    }

    pub fn add<const NTT: bool>(&mut self, r: &Ring<u64>, a: Option<&Poly<u64>>) {
        assert!(NTT, "invalid parameterization: const NTT must be true");
        pack_core::<NTT>(
            r,
            a,
            &mut self.accumulators,
            &self.n_inv,
            &self.x_pow_2,
            &mut self.tmp_a,
            &mut self.tmp_b,
            0,
        );
        self.counter += 1;
        if self.counter == r.n() {
            self.results
                .push(self.accumulators[r.log_n() - 1].buf.clone());
            self.reset();
        }
    }

    pub fn flush<const NTT: bool>(&mut self, r: &Ring<u64>) {
        assert!(NTT, "invalid parameterization: const NTT must be true");
        if self.counter != 0 {
            while self.counter != r.n() - 1 {
                self.add::<NTT>(r, None);
            }
        }
    }
}

fn pack_core<const NTT: bool>(
    r: &Ring<u64>,
    a: Option<&Poly<u64>>,
    accumulators: &mut [Accumulator],
    n_inv: &Barrett<u64>,
    x_pow_2: &[Poly<u64>],
    tmp_a: &mut Poly<u64>,
    tmp_b: &mut Poly<u64>,
    i: usize,
) {
    if i == r.log_n() {
        return;
    }

    let (acc_prev, acc_next) = accumulators.split_at_mut(1);

    if !acc_prev[0].control {
        let acc_mut_ref: &mut Accumulator = &mut acc_prev[0]; // from split_at_mut

        if let Some(a_ref) = a {
            acc_mut_ref.buf.copy_from(a_ref);
            acc_mut_ref.value = true
        } else {
            acc_mut_ref.value = false
        }
        acc_mut_ref.control = true;
    } else {
        combine::<true>(r, &mut acc_prev[0], a, n_inv, x_pow_2, tmp_a, tmp_b, i);
        acc_prev[0].control = false;

        if acc_prev[0].value {
            pack_core::<NTT>(
                r,
                Some(&acc_prev[0].buf),
                acc_next,
                n_inv,
                x_pow_2,
                tmp_a,
                tmp_b,
                i + 1,
            );
        } else {
            pack_core::<NTT>(r, None, acc_next, n_inv, x_pow_2, tmp_a, tmp_b, i + 1);
        }
    }
}

fn combine<const NTT: bool>(
    r: &Ring<u64>,
    acc: &mut Accumulator,
    b: Option<&Poly<u64>>,
    n_inv: &Barrett<u64>,
    x_pow_2: &[Poly<u64>],
    tmp_a: &mut Poly<u64>,
    tmp_b: &mut Poly<u64>,
    i: usize,
) {
    let log_n = r.log_n();
    let log_nth_root = log_n + 1;
    let nth_root = 1 << log_nth_root;
    let gal_el: usize = r.galois_element((1 << i) >> 1, i == 0, log_nth_root);

    let a: &mut Poly<u64> = &mut acc.buf;

    if acc.value {
        if i == 0 {
            r.a_mul_b_scalar_barrett_into_a::<ONCE>(n_inv, a);
        }

        if let Some(b) = b {
            // tmp_a = b * X^t
            r.a_mul_b_montgomery_into_c::<ONCE>(b, &x_pow_2[log_n - i - 1], tmp_a);

            if i == 0 {
                r.a_mul_b_scalar_barrett_into_a::<ONCE>(&n_inv, tmp_a);
            }

            // tmp_b = a - b*X^t
            r.a_sub_b_into_c::<1, ONCE>(a, tmp_a, tmp_b);

            // a = a + b * X^t
            r.a_add_b_into_b::<ONCE>(tmp_a, a);

            // a = a + b * X^t + phi(a - b * X^t)
            r.a_apply_automorphism_add_b_into_b::<ONCE, NTT>(tmp_b, gal_el, nth_root, a);
        } else {
            // tmp_a = phi(a)
            r.a_apply_automorphism_into_b::<NTT>(a, gal_el, nth_root, tmp_a);
            // a = a + phi(a)
            r.a_add_b_into_b::<ONCE>(tmp_a, a);
        }
    } else {
        if let Some(b) = b {
            // tmp_b = b * X^t
            r.a_mul_b_montgomery_into_c::<ONCE>(b, &x_pow_2[log_n - i - 1], tmp_b);

            if i == 0 {
                r.a_mul_b_scalar_barrett_into_a::<ONCE>(&n_inv, tmp_b);
            }

            // tmp_a = phi(b * X^t)
            r.a_apply_automorphism_into_b::<NTT>(tmp_b, gal_el, nth_root, tmp_a);

            // a = (b* X^t - phi(b* X^t))
            r.a_sub_b_into_c::<1, ONCE>(tmp_b, tmp_a, a);
            acc.value = true
        }
    }
}
