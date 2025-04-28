use crate::{Backend, Module, VecZnx, ZnxLayout};
use rand_distr::{Distribution, Normal};
use sampling::source::Source;

pub trait Sampling {
    /// Fills the first `size` size with uniform values in \[-2^{log_base2k-1}, 2^{log_base2k-1}\]
    fn fill_uniform(&self, log_base2k: usize, a: &mut VecZnx, col_i: usize, size: usize, source: &mut Source);

    /// Adds vector sampled according to the provided distribution, scaled by 2^{-log_k} and bounded to \[-bound, bound\].
    fn add_dist_f64<D: Distribution<f64>>(
        &self,
        log_base2k: usize,
        a: &mut VecZnx,
        col_i: usize,
        log_k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    );

    /// Adds a discrete normal vector scaled by 2^{-log_k} with the provided standard deviation and bounded to \[-bound, bound\].
    fn add_normal(
        &self,
        log_base2k: usize,
        a: &mut VecZnx,
        col_i: usize,
        log_k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    );
}

impl<B: Backend> Sampling for Module<B> {
    fn fill_uniform(&self, log_base2k: usize, a: &mut VecZnx, col_i: usize, size: usize, source: &mut Source) {
        let base2k: u64 = 1 << log_base2k;
        let mask: u64 = base2k - 1;
        let base2k_half: i64 = (base2k >> 1) as i64;
        (0..size).for_each(|j| {
            a.at_poly_mut(col_i, j)
                .iter_mut()
                .for_each(|x| *x = (source.next_u64n(base2k, mask) as i64) - base2k_half);
        })
    }

    fn add_dist_f64<D: Distribution<f64>>(
        &self,
        log_base2k: usize,
        a: &mut VecZnx,
        col_i: usize,
        log_k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) {
        assert!(
            (bound.log2().ceil() as i64) < 64,
            "invalid bound: ceil(log2(bound))={} > 63",
            (bound.log2().ceil() as i64)
        );

        let limb: usize = (log_k + log_base2k - 1) / log_base2k - 1;
        let log_base2k_rem: usize = log_k % log_base2k;

        if log_base2k_rem != 0 {
            a.at_poly_mut(col_i, limb).iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a += (dist_f64.round() as i64) << log_base2k_rem;
            });
        } else {
            a.at_poly_mut(col_i, limb).iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a += dist_f64.round() as i64
            });
        }
    }

    fn add_normal(
        &self,
        log_base2k: usize,
        a: &mut VecZnx,
        col_i: usize,
        log_k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) {
        self.add_dist_f64(
            log_base2k,
            a,
            col_i,
            log_k,
            source,
            Normal::new(0.0, sigma).unwrap(),
            bound,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::Sampling;
    use crate::{FFT64, Module, Stats, VecZnx, ZnxBase, ZnxLayout};
    use sampling::source::Source;

    #[test]
    fn fill_uniform() {
        let n: usize = 4096;
        let module: Module<FFT64> = Module::<FFT64>::new(n);
        let log_base2k: usize = 17;
        let size: usize = 5;
        let mut source: Source = Source::new([0u8; 32]);
        let cols: usize = 2;
        let zero: Vec<i64> = vec![0; n];
        let one_12_sqrt: f64 = 0.28867513459481287;
        (0..cols).for_each(|col_i| {
            let mut a: VecZnx = VecZnx::new(&module, cols, size);
            module.fill_uniform(log_base2k, &mut a, col_i, size, &mut source);
            (0..cols).for_each(|col_j| {
                if col_j != col_i {
                    (0..size).for_each(|limb_i| {
                        assert_eq!(a.at_poly(col_j, limb_i), zero);
                    })
                } else {
                    let std: f64 = a.std(col_i, log_base2k);
                    assert!(
                        (std - one_12_sqrt).abs() < 0.01,
                        "std={} ~!= {}",
                        std,
                        one_12_sqrt
                    );
                }
            })
        });
    }

    #[test]
    fn add_normal() {
        let n: usize = 4096;
        let module: Module<FFT64> = Module::<FFT64>::new(n);
        let log_base2k: usize = 17;
        let log_k: usize = 2 * 17;
        let size: usize = 5;
        let sigma: f64 = 3.2;
        let bound: f64 = 6.0 * sigma;
        let mut source: Source = Source::new([0u8; 32]);
        let cols: usize = 2;
        let zero: Vec<i64> = vec![0; n];
        let k_f64: f64 = (1u64 << log_k as u64) as f64;
        (0..cols).for_each(|col_i| {
            let mut a: VecZnx = VecZnx::new(&module, cols, size);
            module.add_normal(log_base2k, &mut a, col_i, log_k, &mut source, sigma, bound);
            (0..cols).for_each(|col_j| {
                if col_j != col_i {
                    (0..size).for_each(|limb_i| {
                        assert_eq!(a.at_poly(col_j, limb_i), zero);
                    })
                } else {
                    let std: f64 = a.std(col_i, log_base2k) * k_f64;
                    assert!((std - sigma).abs() < 0.1, "std={} ~!= {}", std, sigma);
                }
            })
        });
    }
}
