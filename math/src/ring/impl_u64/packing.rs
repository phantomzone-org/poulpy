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
                        self.a_apply_automorphism_add_b_into_b::<ONCE, true>(&tmpa, gal_el, 2 << self.log_n(), poly_lo);
                    } else {
                        self.a_apply_automorphism_into_b::<true>(poly_lo, gal_el, nth_root, &mut tmpa);
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
        assert!(r > l, "invalid input vec: not sorted or collision between indices");
        gap = min(gap, r - l);
        if gap == 1 {
            break;
        }
    }
    gap
}


pub struct StreamRepacker{
    accumulators: Vec<Accumulator>,
    buf0: Poly<u64>,
    buf1: Poly<u64>,
    buf_auto: Poly<u64>,
    x_pow_2: Vec<Poly<Montgomery<u64>>>,
    n_inv: Barrett<u64>,
    counter: usize,
}

pub struct Accumulator{
    buf: [Option<Rc<Poly<u64>>>; 2],
    control: bool,
}

impl Accumulator{
    pub fn new(r: &Ring<u64>) -> Self{
        Self { buf: [Some(Rc::new(r.new_poly())), None], control: false }
    }
}

impl StreamRepacker{
    pub fn new(r: &Ring<u64>) -> Self{

        let mut accumulators: Vec<Accumulator> = Vec::<Accumulator>::new();

        (0..r.log_n()).for_each(|_|
            accumulators.push(Accumulator::new(r))
        );

        Self{
            accumulators: accumulators,
            buf0: r.new_poly(),
            buf1: r.new_poly(),
            buf_auto: r.new_poly(),
            x_pow_2: r.gen_x_pow_2::<true, false>(r.log_n()),
            n_inv: r.modulus.barrett.prepare(r.modulus.inv(r.n() as u64)),
            counter:0,
        }
    }

    fn merge_ab(&mut self, r: &Ring<u64>, a: &Poly<u64>, b: &Poly<u64>, i: usize) -> &Poly<u64>{

        let tmp_a: &mut Poly<u64> = &mut self.buf0;
        let tmp_b: &mut Poly<u64> = &mut self.buf1;

        r.a_mul_b_montgomery_into_c::<ONCE>(a, &self.x_pow_2[r.log_n()-i-1], tmp_a);
        r.a_sub_b_into_c::<1, ONCE>(a, tmp_a, tmp_b);
        r.a_add_b_into_b::<ONCE>(a, tmp_a);

        if i == 0{
            r.a_mul_b_scalar_barrett_into_a::<ONCE>(&self.n_inv, tmp_a);
            r.a_mul_b_scalar_barrett_into_a::<ONCE>(&self.n_inv, tmp_b);
        }

        let log_nth_root = r.log_n()+1;
        let nth_root = 1<<log_nth_root;
            
        let gal_el: usize = r.galois_element((1 << i) >> 1, i == 0, log_nth_root);

        r.a_apply_automorphism_add_b_into_b::<ONCE, true>(tmp_b, gal_el, nth_root, tmp_a);

        tmp_a
    }

    fn merge_a(&mut self, r: &Ring<u64>, a: &Poly<u64>, i: usize) -> &Poly<u64>{

        let tmp_a: &mut Poly<u64> = &mut self.buf0;
 
        let log_nth_root = r.log_n()+1;
        let nth_root = 1<<log_nth_root;
        let gal_el: usize = r.galois_element((1 << i) >> 1, i == 0, log_nth_root);

        if i == 0{
            r.a_mul_b_scalar_barrett_into_a::<ONCE>(&self.n_inv, tmp_a);
            r.a_apply_automorphism_into_b::<true>(tmp_a, gal_el, nth_root, &mut self.buf_auto)
            r.a_add_b_into_c::<ONCE>(&self.buf_auto, a, tmp_a);
        }else{
            r.a_apply_automorphism_into_b::<true>(a, gal_el, nth_root, tmp_a);
            r.a_add_b_into_b::<ONCE>(a, tmp_a);
        }

        tmp_a
    }

    fn merge_b(&mut self, r: &Ring<u64>, b: &Poly<u64>, i: usize) -> &Poly<u64>{

        let tmp_a: &mut Poly<u64> = &mut self.buf0;
        let tmp_b: &mut Poly<u64> = &mut self.buf1;

        let log_nth_root = r.log_n()+1;
        let nth_root = 1<<log_nth_root;
        let gal_el: usize = r.galois_element((1 << i) >> 1, i == 0, log_nth_root);

        if i == 0{
            r.a_mul_b_scalar_barrett_into_c::<ONCE>(&self.n_inv, b, tmp_b);
            r.a_mul_b_montgomery_into_a::<ONCE>(&self.x_pow_2[r.log_n()-i-1], tmp_b);
        }else{
            r.a_mul_b_montgomery_into_c::<ONCE>(b, &self.x_pow_2[r.log_n()-i-1], tmp_b);
        }

        r.a_apply_automorphism_into_b::<true>(tmp_b, gal_el, nth_root, &mut self.buf_auto);
        r.a_sub_b_into_a::<1, ONCE>(&self.buf_auto, tmp_b);

        tmp_b
    }
}