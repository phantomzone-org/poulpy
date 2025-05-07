use crate::znx_base::ZnxViewMut;
use crate::{FFT64, VecZnx, VecZnxBig, VecZnxBigToMut, VecZnxToMut};
use rand_distr::{Distribution, Normal};
use sampling::source::Source;

pub trait FillUniform {
    /// Fills the first `size` size with uniform values in \[-2^{log_base2k-1}, 2^{log_base2k-1}\]
    fn fill_uniform(&mut self, log_base2k: usize, col_i: usize, size: usize, source: &mut Source);
}

pub trait FillDistF64 {
    fn fill_dist_f64<D: Distribution<f64>>(
        &mut self,
        log_base2k: usize,
        col_i: usize,
        log_k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    );
}

pub trait AddDistF64 {
    /// Adds vector sampled according to the provided distribution, scaled by 2^{-log_k} and bounded to \[-bound, bound\].
    fn add_dist_f64<D: Distribution<f64>>(
        &mut self,
        log_base2k: usize,
        col_i: usize,
        log_k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    );
}

pub trait FillNormal {
    fn fill_normal(&mut self, log_base2k: usize, col_i: usize, log_k: usize, source: &mut Source, sigma: f64, bound: f64);
}

pub trait AddNormal {
    /// Adds a discrete normal vector scaled by 2^{-log_k} with the provided standard deviation and bounded to \[-bound, bound\].
    fn add_normal(&mut self, log_base2k: usize, col_i: usize, log_k: usize, source: &mut Source, sigma: f64, bound: f64);
}

impl<T> FillUniform for VecZnx<T>
where
    VecZnx<T>: VecZnxToMut,
{
    fn fill_uniform(&mut self, log_base2k: usize, col_i: usize, size: usize, source: &mut Source) {
        let mut a: VecZnx<&mut [u8]> = self.to_mut();
        let base2k: u64 = 1 << log_base2k;
        let mask: u64 = base2k - 1;
        let base2k_half: i64 = (base2k >> 1) as i64;
        (0..size).for_each(|j| {
            a.at_mut(col_i, j)
                .iter_mut()
                .for_each(|x| *x = (source.next_u64n(base2k, mask) as i64) - base2k_half);
        })
    }
}

impl<T> FillDistF64 for VecZnx<T>
where
    VecZnx<T>: VecZnxToMut,
{
    fn fill_dist_f64<D: Distribution<f64>>(
        &mut self,
        log_base2k: usize,
        col_i: usize,
        log_k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) {
        let mut a: VecZnx<&mut [u8]> = self.to_mut();
        assert!(
            (bound.log2().ceil() as i64) < 64,
            "invalid bound: ceil(log2(bound))={} > 63",
            (bound.log2().ceil() as i64)
        );

        let limb: usize = (log_k + log_base2k - 1) / log_base2k - 1;
        let log_base2k_rem: usize = log_k % log_base2k;

        if log_base2k_rem != 0 {
            a.at_mut(col_i, limb).iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a = (dist_f64.round() as i64) << log_base2k_rem;
            });
        } else {
            a.at_mut(col_i, limb).iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a = dist_f64.round() as i64
            });
        }
    }
}

impl<T> AddDistF64 for VecZnx<T>
where
    VecZnx<T>: VecZnxToMut,
{
    fn add_dist_f64<D: Distribution<f64>>(
        &mut self,
        log_base2k: usize,
        col_i: usize,
        log_k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) {
        let mut a: VecZnx<&mut [u8]> = self.to_mut();
        assert!(
            (bound.log2().ceil() as i64) < 64,
            "invalid bound: ceil(log2(bound))={} > 63",
            (bound.log2().ceil() as i64)
        );

        let limb: usize = (log_k + log_base2k - 1) / log_base2k - 1;
        let log_base2k_rem: usize = log_k % log_base2k;

        if log_base2k_rem != 0 {
            a.at_mut(col_i, limb).iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a += (dist_f64.round() as i64) << log_base2k_rem;
            });
        } else {
            a.at_mut(col_i, limb).iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a += dist_f64.round() as i64
            });
        }
    }
}

impl<T> FillNormal for VecZnx<T>
where
    VecZnx<T>: VecZnxToMut,
{
    fn fill_normal(&mut self, log_base2k: usize, col_i: usize, log_k: usize, source: &mut Source, sigma: f64, bound: f64) {
        self.fill_dist_f64(
            log_base2k,
            col_i,
            log_k,
            source,
            Normal::new(0.0, sigma).unwrap(),
            bound,
        );
    }
}

impl<T> AddNormal for VecZnx<T>
where
    VecZnx<T>: VecZnxToMut,
{
    fn add_normal(&mut self, log_base2k: usize, col_i: usize, log_k: usize, source: &mut Source, sigma: f64, bound: f64) {
        self.add_dist_f64(
            log_base2k,
            col_i,
            log_k,
            source,
            Normal::new(0.0, sigma).unwrap(),
            bound,
        );
    }
}

impl<T> FillDistF64 for VecZnxBig<T, FFT64>
where
    VecZnxBig<T, FFT64>: VecZnxBigToMut<FFT64>,
{
    fn fill_dist_f64<D: Distribution<f64>>(
        &mut self,
        log_base2k: usize,
        col_i: usize,
        log_k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) {
        let mut a: VecZnxBig<&mut [u8], FFT64> = self.to_mut();
        assert!(
            (bound.log2().ceil() as i64) < 64,
            "invalid bound: ceil(log2(bound))={} > 63",
            (bound.log2().ceil() as i64)
        );

        let limb: usize = (log_k + log_base2k - 1) / log_base2k - 1;
        let log_base2k_rem: usize = log_k % log_base2k;

        if log_base2k_rem != 0 {
            a.at_mut(col_i, limb).iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a = (dist_f64.round() as i64) << log_base2k_rem;
            });
        } else {
            a.at_mut(col_i, limb).iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a = dist_f64.round() as i64
            });
        }
    }
}

impl<T> AddDistF64 for VecZnxBig<T, FFT64>
where
    VecZnxBig<T, FFT64>: VecZnxBigToMut<FFT64>,
{
    fn add_dist_f64<D: Distribution<f64>>(
        &mut self,
        log_base2k: usize,
        col_i: usize,
        log_k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) {
        let mut a: VecZnxBig<&mut [u8], FFT64> = self.to_mut();
        assert!(
            (bound.log2().ceil() as i64) < 64,
            "invalid bound: ceil(log2(bound))={} > 63",
            (bound.log2().ceil() as i64)
        );

        let limb: usize = (log_k + log_base2k - 1) / log_base2k - 1;
        let log_base2k_rem: usize = log_k % log_base2k;

        if log_base2k_rem != 0 {
            a.at_mut(col_i, limb).iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a += (dist_f64.round() as i64) << log_base2k_rem;
            });
        } else {
            a.at_mut(col_i, limb).iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a += dist_f64.round() as i64
            });
        }
    }
}

impl<T> FillNormal for VecZnxBig<T, FFT64>
where
    VecZnxBig<T, FFT64>: VecZnxBigToMut<FFT64>,
{
    fn fill_normal(&mut self, log_base2k: usize, col_i: usize, log_k: usize, source: &mut Source, sigma: f64, bound: f64) {
        self.fill_dist_f64(
            log_base2k,
            col_i,
            log_k,
            source,
            Normal::new(0.0, sigma).unwrap(),
            bound,
        );
    }
}

impl<T> AddNormal for VecZnxBig<T, FFT64>
where
    VecZnxBig<T, FFT64>: VecZnxBigToMut<FFT64>,
{
    fn add_normal(&mut self, log_base2k: usize, col_i: usize, log_k: usize, source: &mut Source, sigma: f64, bound: f64) {
        self.add_dist_f64(
            log_base2k,
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
    use super::{AddNormal, FillUniform};
    use crate::vec_znx_ops::*;
    use crate::znx_base::*;
    use crate::{FFT64, Module, Stats, VecZnx};
    use sampling::source::Source;

    #[test]
    fn vec_znx_fill_uniform() {
        let n: usize = 4096;
        let module: Module<FFT64> = Module::<FFT64>::new(n);
        let log_base2k: usize = 17;
        let size: usize = 5;
        let mut source: Source = Source::new([0u8; 32]);
        let cols: usize = 2;
        let zero: Vec<i64> = vec![0; n];
        let one_12_sqrt: f64 = 0.28867513459481287;
        (0..cols).for_each(|col_i| {
            let mut a: VecZnx<_> = module.new_vec_znx(cols, size);
            a.fill_uniform(log_base2k, col_i, size, &mut source);
            (0..cols).for_each(|col_j| {
                if col_j != col_i {
                    (0..size).for_each(|limb_i| {
                        assert_eq!(a.at(col_j, limb_i), zero);
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
    fn vec_znx_add_normal() {
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
            let mut a: VecZnx<_> = module.new_vec_znx(cols, size);
            a.add_normal(log_base2k, col_i, log_k, &mut source, sigma, bound);
            (0..cols).for_each(|col_j| {
                if col_j != col_i {
                    (0..size).for_each(|limb_i| {
                        assert_eq!(a.at(col_j, limb_i), zero);
                    })
                } else {
                    let std: f64 = a.std(col_i, log_base2k) * k_f64;
                    assert!((std - sigma).abs() < 0.1, "std={} ~!= {}", std, sigma);
                }
            })
        });
    }
}
