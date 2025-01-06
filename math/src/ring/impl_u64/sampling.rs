use crate::modulus::WordOps;
use crate::poly::{Poly, PolyRNS};
use crate::ring::{Ring, RingRNS};
use sampling::source::Source;

impl Ring<u64> {
    pub fn fill_uniform(&self, source: &mut Source, a: &mut Poly<u64>) {
        let max: u64 = self.modulus.q;
        let mask: u64 = max.mask();
        a.0.iter_mut()
            .for_each(|a| *a = source.next_u64n(max, mask));
    }
}

impl RingRNS<u64> {
    pub fn fill_uniform(&self, source: &mut Source, a: &mut PolyRNS<u64>) {
        self.0
            .iter()
            .enumerate()
            .for_each(|(i, r)| r.fill_uniform(source, a.at_mut(i)));
    }
}
