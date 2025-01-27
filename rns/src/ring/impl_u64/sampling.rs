use crate::modulus::WordOps;
use crate::poly::{Poly, PolyRNS};
use crate::ring::{Ring, RingRNS};
use num::ToPrimitive;
use rand_distr::{Distribution, Normal};
use sampling::source::Source;

impl Ring<u64> {
    pub fn fill_uniform(&self, source: &mut Source, a: &mut Poly<u64>) {
        let max: u64 = self.modulus.q;
        let mask: u64 = max.mask();
        a.0.iter_mut()
            .for_each(|a| *a = source.next_u64n(max, mask));
    }

    pub fn fill_dist_f64<T: Distribution<f64>>(
        &self,
        source: &mut Source,
        dist: T,
        bound: f64,
        a: &mut Poly<u64>,
    ) {
        let max: u64 = self.modulus.q;
        a.0.iter_mut().for_each(|a| {
            let mut dist_f64: f64 = dist.sample(source);

            while dist_f64.abs() > bound {
                dist_f64 = dist.sample(source)
            }

            let dist_u64: u64 = (dist_f64 + 0.5).abs().to_u64().unwrap();
            let sign: u64 = dist_f64.to_bits() >> 63;

            *a = (dist_u64 * sign) | (max - dist_u64) * (sign ^ 1)
        });
    }

    pub fn fill_normal(&self, source: &mut Source, sigma: f64, bound: f64, a: &mut Poly<u64>) {
        self.fill_dist_f64(source, Normal::new(0.0, sigma).unwrap(), bound, a);
    }
}

impl RingRNS<u64> {
    pub fn fill_uniform(&self, source: &mut Source, a: &mut PolyRNS<u64>) {
        self.0
            .iter()
            .enumerate()
            .for_each(|(i, r)| r.fill_uniform(source, a.at_mut(i)));
    }

    pub fn fill_dist_f64<T: Distribution<f64>>(
        &self,
        source: &mut Source,
        dist: T,
        bound: f64,
        a: &mut PolyRNS<u64>,
    ) {
        (0..a.n()).for_each(|j| {
            let mut dist_f64: f64 = dist.sample(source);

            while dist_f64.abs() > bound {
                dist_f64 = dist.sample(source)
            }

            let dist_u64: u64 = (dist_f64 + 0.5).abs().to_u64().unwrap();
            let sign: u64 = dist_f64.to_bits() >> 63;

            self.0.iter().enumerate().for_each(|(i, r)| {
                a.at_mut(i).0[j] = (dist_u64 * sign) | (r.modulus.q - dist_u64) * (sign ^ 1);
            })
        })
    }
}
