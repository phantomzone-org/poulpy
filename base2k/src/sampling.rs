use crate::{Infos, VecZnx};
use rand_distr::{Distribution, Normal};
use sampling::source::Source;

pub trait Sampling {
    /// Fills the first `limbs` limbs with uniform values in \[-2^{log_base2k-1}, 2^{log_base2k-1}\]
    fn fill_uniform(&mut self, log_base2k: usize, limbs: usize, source: &mut Source);

    /// Adds vector sampled according to the provided distribution, scaled by 2^{-log_k} and bounded to \[-bound, bound\].
    fn add_dist_f64<T: Distribution<f64>>(
        &mut self,
        log_base2k: usize,
        log_k: usize,
        source: &mut Source,
        dist: T,
        bound: f64,
    );

    /// Adds a discrete normal vector scaled by 2^{-log_k} with the provided standard deviation and bounded to \[-bound, bound\].
    fn add_normal(
        &mut self,
        log_base2k: usize,
        log_k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    );
}

impl Sampling for VecZnx {
    fn fill_uniform(&mut self, log_base2k: usize, limbs: usize, source: &mut Source) {
        let base2k: u64 = 1 << log_base2k;
        let mask: u64 = base2k - 1;
        let base2k_half: i64 = (base2k >> 1) as i64;

        let size: usize = self.n() * (limbs - 1);

        self.data[..size]
            .iter_mut()
            .for_each(|x| *x = (source.next_u64n(base2k, mask) as i64) - base2k_half);
    }

    fn add_dist_f64<T: Distribution<f64>>(
        &mut self,
        log_base2k: usize,
        log_k: usize,
        source: &mut Source,
        dist: T,
        bound: f64,
    ) {
        assert!(
            (bound.log2().ceil() as i64) < 64,
            "invalid bound: ceil(log2(bound))={} > 63",
            (bound.log2().ceil() as i64)
        );

        let log_base2k_rem: usize = log_k % log_base2k;

        if log_base2k_rem != 0 {
            self.at_mut(self.limbs() - 1).iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a += (dist_f64.round() as i64) << log_base2k_rem
            });
        } else {
            self.at_mut(self.limbs() - 1).iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a += dist_f64.round() as i64
            });
        }
    }

    fn add_normal(
        &mut self,
        log_base2k: usize,
        log_k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) {
        self.add_dist_f64(
            log_base2k,
            log_k,
            source,
            Normal::new(0.0, sigma).unwrap(),
            bound,
        );
    }
}
