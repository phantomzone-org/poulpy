use crate::ring::{Ring, RingRNS};
use crate::poly::PolyRNS;
use crate::modulus::montgomery::Montgomery;
use crate::modulus::barrett::Barrett;
use crate::modulus::REDUCEMOD;
use num_bigint::BigInt;

pub fn new_rings(n: usize, moduli: Vec<u64>) -> Vec<Ring<u64>>{
    assert!(!moduli.is_empty(), "moduli cannot be empty");
    let rings: Vec<Ring<u64>> = moduli
        .into_iter()
        .map(|prime| Ring::new(n, prime, 1)) 
        .collect();
    return rings
}

impl<'a> RingRNS<'a, u64>{
    pub fn new(rings:&'a [Ring<u64>]) -> Self{
        RingRNS(rings)
    }

    pub fn rescaling_constant(&self) -> Vec<Barrett<u64>> {
        let level = self.level();
        let q_scale: u64 = self.0[level].modulus.q;
        (0..level).map(|i| {self.0[i].modulus.barrett.prepare(self.0[i].modulus.q - self.0[i].modulus.inv(q_scale))}).collect()
    }

    pub fn set_poly_from_bigint(&self, coeffs: &[BigInt], step:usize, a: &mut PolyRNS<u64>){
        let level = self.level();
        assert!(level <= a.level(), "invalid level: level={} > a.level()={}", level, a.level());
        (0..level).for_each(|i|{self.0[i].from_bigint(coeffs, step, a.at_mut(i))});
    }
}

impl RingRNS<'_, u64>{
    
    #[inline(always)]
    pub fn add<const REDUCE: REDUCEMOD>(&self, a: &PolyRNS<u64>, b: &PolyRNS<u64>, c: &mut PolyRNS<u64>){
        let level: usize = self.level();
        debug_assert!(self.max_level() <= level, "max_level={} < level={}", self.max_level(), level);
        debug_assert!(a.level() >= level, "a.level()={} < level={}", a.level(), level);
        debug_assert!(b.level() >= level, "b.level()={} < level={}", b.level(), level);
        debug_assert!(c.level() >= level, "c.level()={} < level={}", c.level(), level);
        self.0.iter().take(level + 1).enumerate().for_each(|(i, ring)| ring.add::<REDUCE>(&a.0[i], &b.0[i], &mut c.0[i]));
    }

    #[inline(always)]
    pub fn add_inplace<const REDUCE: REDUCEMOD>(&self, a: &PolyRNS<u64>, b: &mut PolyRNS<u64>){
        let level: usize = self.level();
        debug_assert!(self.max_level() <= level, "max_level={} < level={}", self.max_level(), level);
        debug_assert!(a.level() >= level, "a.level()={} < level={}", a.level(), level);
        debug_assert!(b.level() >= level, "b.level()={} < level={}", b.level(), level);
        self.0.iter().take(level + 1).enumerate().for_each(|(i, ring)| ring.add_inplace::<REDUCE>(&a.0[i], &mut b.0[i]));
    }

    #[inline(always)]
    pub fn sub<const REDUCE: REDUCEMOD>(&self, a: &PolyRNS<u64>, b: &PolyRNS<u64>, c: &mut PolyRNS<u64>){
        let level: usize = self.level();
        debug_assert!(self.max_level() <= level, "max_level={} < level={}", self.max_level(), level);
        debug_assert!(a.level() >= level, "a.level()={} < level={}", a.level(), level);
        debug_assert!(b.level() >= level, "b.level()={} < level={}", b.level(), level);
        debug_assert!(c.level() >= level, "c.level()={} < level={}", c.level(), level);
        self.0.iter().take(level + 1).enumerate().for_each(|(i, ring)| ring.sub::<REDUCE>(&a.0[i], &b.0[i], &mut c.0[i]));
    }

    #[inline(always)]
    pub fn sub_inplace<const REDUCE: REDUCEMOD>(&self, a: &PolyRNS<u64>, b: &mut PolyRNS<u64>){
        let level: usize = self.level();
        debug_assert!(self.max_level() <= level, "max_level={} < level={}", self.max_level(), level);
        debug_assert!(a.level() >= level, "a.level()={} < level={}", a.level(), level);
        debug_assert!(b.level() >= level, "b.level()={} < level={}", b.level(), level);
        self.0.iter().take(level + 1).enumerate().for_each(|(i, ring)| ring.sub_inplace::<REDUCE>(&a.0[i], &mut b.0[i]));
    }

    #[inline(always)]
    pub fn neg<const REDUCE: REDUCEMOD>(&self, a: &PolyRNS<u64>, b: &mut PolyRNS<u64>){
        let level: usize = self.level();
        debug_assert!(self.max_level() <= level, "max_level={} < level={}", self.max_level(), level);
        debug_assert!(a.level() >= level, "a.level()={} < level={}", a.level(), level);
        debug_assert!(b.level() >= level, "b.level()={} < level={}", b.level(), level);
        self.0.iter().take(level + 1).enumerate().for_each(|(i, ring)| ring.neg::<REDUCE>(&a.0[i], &mut b.0[i]));
    }

    #[inline(always)]
    pub fn neg_inplace<const REDUCE: REDUCEMOD>(&self, a: &mut PolyRNS<u64>){
        let level: usize = self.level();
        debug_assert!(self.max_level() <= level, "max_level={} < level={}", self.max_level(), level);
        debug_assert!(a.level() >= level, "a.level()={} < level={}", a.level(), level);
        self.0.iter().take(level + 1).enumerate().for_each(|(i, ring)| ring.neg_inplace::<REDUCE>(&mut a.0[i]));
    }

    #[inline(always)]
    pub fn  mul_montgomery_external<const REDUCE:REDUCEMOD>(&self, a:&PolyRNS<Montgomery<u64>>, b:&PolyRNS<u64>, c: &mut PolyRNS<u64>){
        let level: usize = self.level();
        debug_assert!(self.max_level() <= level, "max_level={} < level={}", self.max_level(), level);
        debug_assert!(a.level() >= level, "a.level()={} < level={}", a.level(), level);
        debug_assert!(b.level() >= level, "b.level()={} < level={}", b.level(), level);
        debug_assert!(c.level() >= level, "c.level()={} < level={}", c.level(), level);
        self.0.iter().take(level + 1).enumerate().for_each(|(i, ring)| ring.mul_montgomery_external::<REDUCE>(&a.0[i], &b.0[i], &mut c.0[i]));
    }

    #[inline(always)]
    pub fn mul_montgomery_external_inplace<const REDUCE:REDUCEMOD>(&self, a:&PolyRNS<Montgomery<u64>>, b:&mut PolyRNS<u64>){
        let level: usize = self.level();
        debug_assert!(self.max_level() <= level, "max_level={} < level={}", self.max_level(), level);
        debug_assert!(a.level() >= level, "a.level()={} < level={}", a.level(), level);
        debug_assert!(b.level() >= level, "b.level()={} < level={}", b.level(), level);
        self.0.iter().take(level + 1).enumerate().for_each(|(i, ring)| ring.mul_montgomery_external_inplace::<REDUCE>(&a.0[i], &mut b.0[i]));
    }
}