use crate::ring::{Ring, RingRNS};
use crate::poly::PolyRNS;
use crate::modulus::montgomery::Montgomery;
use crate::modulus::REDUCEMOD;

impl RingRNS<u64>{
    pub fn new(n:usize, moduli: Vec<u64>) -> Self{
        assert!(!moduli.is_empty(), "moduli cannot be empty");
        let rings: Vec<Ring<u64>> = moduli
            .into_iter()
            .map(|prime: u64| Ring::new(n, prime, 1))
            .collect();
        RingRNS(rings)
    }

    pub fn n(&self) -> usize{
        self.0[0].n()
    }

    pub fn max_level(&self) -> usize{
        self.0.len()-1
    }
}

impl RingRNS<u64>{
    
}

impl RingRNS<u64>{
    
    #[inline(always)]
    pub fn add<const LEVEL:usize, const REDUCE: REDUCEMOD>(&self, a: &PolyRNS<u64>, b: &PolyRNS<u64>, c: &mut PolyRNS<u64>){
        debug_assert!(self.max_level() <= LEVEL, "max_level={} < LEVEL={}", self.max_level(), LEVEL);
        debug_assert!(a.n() >= self.n(), "a.n()={} < n={}", a.n(), self.n());
        debug_assert!(b.n() >= self.n(), "b.n()={} < n={}", b.n(), self.n());
        debug_assert!(c.n() >= self.n(), "c.n()={} < n={}", c.n(), self.n());
        debug_assert!(a.level() >= LEVEL, "a.level()={} < LEVEL={}", a.level(), LEVEL);
        debug_assert!(b.level() >= LEVEL, "b.level()={} < LEVEL={}", b.level(), LEVEL);
        debug_assert!(c.level() >= LEVEL, "c.level()={} < LEVEL={}", c.level(), LEVEL);
        self.0.iter().take(LEVEL + 1).enumerate().for_each(|(i, ring)| ring.add::<REDUCE>(&a.0[i], &b.0[i], &mut c.0[i]));
    }

    #[inline(always)]
    pub fn add_inplace<const LEVEL:usize, const REDUCE: REDUCEMOD>(&self, a: &PolyRNS<u64>, b: &mut PolyRNS<u64>){
        debug_assert!(self.max_level() <= LEVEL, "max_level={} < LEVEL={}", self.max_level(), LEVEL);
        debug_assert!(a.n() >= self.n(), "a.n()={} < n={}", a.n(), self.n());
        debug_assert!(b.n() >= self.n(), "b.n()={} < n={}", b.n(), self.n());
        debug_assert!(a.level() >= LEVEL, "a.level()={} < LEVEL={}", a.level(), LEVEL);
        debug_assert!(b.level() >= LEVEL, "b.level()={} < LEVEL={}", b.level(), LEVEL);
        self.0.iter().take(LEVEL + 1).enumerate().for_each(|(i, ring)| ring.add_inplace::<REDUCE>(&a.0[i], &mut b.0[i]));
    }

    #[inline(always)]
    pub fn sub<const LEVEL:usize, const REDUCE: REDUCEMOD>(&self, a: &PolyRNS<u64>, b: &PolyRNS<u64>, c: &mut PolyRNS<u64>){
        debug_assert!(self.max_level() <= LEVEL, "max_level={} < LEVEL={}", self.max_level(), LEVEL);
        debug_assert!(a.n() >= self.n(), "a.n()={} < n={}", a.n(), self.n());
        debug_assert!(b.n() >= self.n(), "b.n()={} < n={}", b.n(), self.n());
        debug_assert!(c.n() >= self.n(), "c.n()={} < n={}", c.n(), self.n());
        debug_assert!(a.level() >= LEVEL, "a.level()={} < LEVEL={}", a.level(), LEVEL);
        debug_assert!(b.level() >= LEVEL, "b.level()={} < LEVEL={}", b.level(), LEVEL);
        debug_assert!(c.level() >= LEVEL, "c.level()={} < LEVEL={}", c.level(), LEVEL);
        self.0.iter().take(LEVEL + 1).enumerate().for_each(|(i, ring)| ring.sub::<REDUCE>(&a.0[i], &b.0[i], &mut c.0[i]));
    }

    #[inline(always)]
    pub fn sub_inplace<const LEVEL:usize, const REDUCE: REDUCEMOD>(&self, a: &PolyRNS<u64>, b: &mut PolyRNS<u64>){
        debug_assert!(self.max_level() <= LEVEL, "max_level={} < LEVEL={}", self.max_level(), LEVEL);
        debug_assert!(a.n() >= self.n(), "a.n()={} < n={}", a.n(), self.n());
        debug_assert!(b.n() >= self.n(), "b.n()={} < n={}", b.n(), self.n());
        debug_assert!(a.level() >= LEVEL, "a.level()={} < LEVEL={}", a.level(), LEVEL);
        debug_assert!(b.level() >= LEVEL, "b.level()={} < LEVEL={}", b.level(), LEVEL);
        self.0.iter().take(LEVEL + 1).enumerate().for_each(|(i, ring)| ring.sub_inplace::<REDUCE>(&a.0[i], &mut b.0[i]));
    }

    #[inline(always)]
    pub fn neg<const LEVEL:usize, const REDUCE: REDUCEMOD>(&self, a: &PolyRNS<u64>, b: &mut PolyRNS<u64>){
        debug_assert!(self.max_level() <= LEVEL, "max_level={} < LEVEL={}", self.max_level(), LEVEL);
        debug_assert!(a.n() >= self.n(), "a.n()={} < n={}", a.n(), self.n());
        debug_assert!(b.n() >= self.n(), "b.n()={} < n={}", b.n(), self.n());
        debug_assert!(a.level() >= LEVEL, "a.level()={} < LEVEL={}", a.level(), LEVEL);
        debug_assert!(b.level() >= LEVEL, "b.level()={} < LEVEL={}", b.level(), LEVEL);
        self.0.iter().take(LEVEL + 1).enumerate().for_each(|(i, ring)| ring.neg::<REDUCE>(&a.0[i], &mut b.0[i]));
    }

    #[inline(always)]
    pub fn neg_inplace<const LEVEL:usize, const REDUCE: REDUCEMOD>(&self, a: &mut PolyRNS<u64>){
        debug_assert!(self.max_level() <= LEVEL, "max_level={} < LEVEL={}", self.max_level(), LEVEL);
        debug_assert!(a.n() >= self.n(), "a.n()={} < n={}", a.n(), self.n());
        debug_assert!(a.level() >= LEVEL, "a.level()={} < LEVEL={}", a.level(), LEVEL);
        self.0.iter().take(LEVEL + 1).enumerate().for_each(|(i, ring)| ring.neg_inplace::<REDUCE>(&mut a.0[i]));
    }

    #[inline(always)]
    pub fn  mul_montgomery_external<const LEVEL:usize, const REDUCE:REDUCEMOD>(&self, a:&PolyRNS<Montgomery<u64>>, b:&PolyRNS<u64>, c: &mut PolyRNS<u64>){
        debug_assert!(self.max_level() <= LEVEL, "max_level={} < LEVEL={}", self.max_level(), LEVEL);
        debug_assert!(a.n() >= self.n(), "a.n()={} < n={}", a.n(), self.n());
        debug_assert!(b.n() >= self.n(), "b.n()={} < n={}", b.n(), self.n());
        debug_assert!(c.n() >= self.n(), "c.n()={} < n={}", c.n(), self.n());
        debug_assert!(a.level() >= LEVEL, "a.level()={} < LEVEL={}", a.level(), LEVEL);
        debug_assert!(b.level() >= LEVEL, "b.level()={} < LEVEL={}", b.level(), LEVEL);
        debug_assert!(c.level() >= LEVEL, "c.level()={} < LEVEL={}", c.level(), LEVEL);
        self.0.iter().take(LEVEL + 1).enumerate().for_each(|(i, ring)| ring.mul_montgomery_external::<REDUCE>(&a.0[i], &b.0[i], &mut c.0[i]));
    }

    #[inline(always)]
    pub fn mul_montgomery_external_inplace<const LEVEL:usize, const REDUCE:REDUCEMOD>(&self, a:&PolyRNS<Montgomery<u64>>, b:&mut PolyRNS<u64>){
        debug_assert!(self.max_level() <= LEVEL, "max_level={} < LEVEL={}", self.max_level(), LEVEL);
        debug_assert!(a.n() >= self.n(), "a.n()={} < n={}", a.n(), self.n());
        debug_assert!(b.n() >= self.n(), "b.n()={} < n={}", b.n(), self.n());
        debug_assert!(a.level() >= LEVEL, "a.level()={} < LEVEL={}", a.level(), LEVEL);
        debug_assert!(b.level() >= LEVEL, "b.level()={} < LEVEL={}", b.level(), LEVEL);
        self.0.iter().take(LEVEL + 1).enumerate().for_each(|(i, ring)| ring.mul_montgomery_external_inplace::<REDUCE>(&a.0[i], &mut b.0[i]));
    }
}