use crate::ring::{Ring, RingRNS};
use crate::poly::PolyRNS;
use crate::modulus::montgomery::Montgomery;
use crate::modulus::barrett::BarrettRNS;
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

    pub fn modulus(&self) -> BigInt{
        let mut modulus = BigInt::from(1);
        self.0.iter().enumerate().for_each(|(_, r)|modulus *= BigInt::from(r.modulus.q));
        modulus
    }

    pub fn rescaling_constant(&self) -> BarrettRNS<u64> {
        let level = self.level();
        let q_scale: u64 = self.0[level].modulus.q;
        BarrettRNS((0..level).map(|i| {self.0[i].modulus.barrett.prepare(self.0[i].modulus.q - self.0[i].modulus.inv(q_scale))}).collect())
    }

    pub fn from_bigint_inplace(&self, coeffs: &[BigInt], step:usize, a: &mut PolyRNS<u64>){
        let level = self.level();
        assert!(level <= a.level(), "invalid level: level={} > a.level()={}", level, a.level());
        (0..level).for_each(|i|{self.0[i].from_bigint(coeffs, step, a.at_mut(i))});
    }

    pub fn to_bigint_inplace(&self, a: &PolyRNS<u64>, step: usize, coeffs: &mut [BigInt]){
        assert!(step <= a.n(), "invalid step: step={} > a.n()={}", step, a.n());
        assert!(coeffs.len() <= a.n() / step, "invalid coeffs: coeffs.len()={} > a.n()/step={}", coeffs.len(), a.n()/step);

        let mut inv_crt: Vec<BigInt> = vec![BigInt::default(); self.level()+1];
        let q_big: BigInt = self.modulus();
        let q_big_half: BigInt = &q_big>>1;

        inv_crt.iter_mut().enumerate().for_each(|(i, a)|{
            let qi_big = BigInt::from(self.0[i].modulus.q);
            *a = &q_big / &qi_big;
            *a *= a.modinv(&qi_big).unwrap();
        });

        (0..self.n()).step_by(step).enumerate().for_each(|(i, j)|{
            coeffs[j] = BigInt::from(a.at(0).0[i]) * &inv_crt[0];
            (1..self.level()+1).for_each(|k|{
                coeffs[j] += BigInt::from(a.at(k).0[i] * &inv_crt[k]);
            });
            coeffs[j] %= &q_big;
            if &coeffs[j] >= &q_big_half{
                coeffs[j] -= &q_big;
            }
        });
    }
}

impl RingRNS<'_, u64>{
    pub fn ntt_inplace<const LAZY:bool>(&self, a: &mut PolyRNS<u64>){
        self.0.iter().enumerate().for_each(|(i, ring)| ring.ntt_inplace::<LAZY>(&mut a.0[i]));
    }

    pub fn intt_inplace<const LAZY:bool>(&self, a: &mut PolyRNS<u64>){
        self.0.iter().enumerate().for_each(|(i, ring)| ring.intt_inplace::<LAZY>(&mut a.0[i]));
    }

    pub fn ntt<const LAZY:bool>(&self, a: &PolyRNS<u64>, b: &mut PolyRNS<u64>){
        self.0.iter().enumerate().for_each(|(i, ring)| ring.ntt::<LAZY>(&a.0[i], &mut b.0[i]));
    }

    pub fn intt<const LAZY:bool>(&self, a: &PolyRNS<u64>, b: &mut PolyRNS<u64>){
        self.0.iter().enumerate().for_each(|(i, ring)| ring.intt::<LAZY>(&a.0[i], &mut b.0[i]));
    }
}

impl RingRNS<'_, u64>{
    
    #[inline(always)]
    pub fn add<const REDUCE: REDUCEMOD>(&self, a: &PolyRNS<u64>, b: &PolyRNS<u64>, c: &mut PolyRNS<u64>){
        debug_assert!(a.level() >= self.level(), "a.level()={} < self.level()={}", a.level(), self.level());
        debug_assert!(b.level() >= self.level(), "b.level()={} < self.level()={}", b.level(), self.level());
        debug_assert!(c.level() >= self.level(), "c.level()={} < self.level()={}", c.level(), self.level());
        self.0.iter().enumerate().for_each(|(i, ring)| ring.add::<REDUCE>(&a.0[i], &b.0[i], &mut c.0[i]));
    }

    #[inline(always)]
    pub fn add_inplace<const REDUCE: REDUCEMOD>(&self, a: &PolyRNS<u64>, b: &mut PolyRNS<u64>){
        debug_assert!(a.level() >= self.level(), "a.level()={} < self.level()={}", a.level(), self.level());
        debug_assert!(b.level() >= self.level(), "b.level()={} < self.level()={}", b.level(), self.level());
        self.0.iter().enumerate().for_each(|(i, ring)| ring.add_inplace::<REDUCE>(&a.0[i], &mut b.0[i]));
    }

    #[inline(always)]
    pub fn sub<const REDUCE: REDUCEMOD>(&self, a: &PolyRNS<u64>, b: &PolyRNS<u64>, c: &mut PolyRNS<u64>){
        debug_assert!(a.level() >= self.level(), "a.level()={} < self.level()={}", a.level(), self.level());
        debug_assert!(b.level() >= self.level(), "b.level()={} < self.level()={}", b.level(), self.level());
        debug_assert!(c.level() >= self.level(), "c.level()={} < self.level()={}", c.level(), self.level());
        self.0.iter().enumerate().for_each(|(i, ring)| ring.sub::<REDUCE>(&a.0[i], &b.0[i], &mut c.0[i]));
    }

    #[inline(always)]
    pub fn sub_inplace<const REDUCE: REDUCEMOD>(&self, a: &PolyRNS<u64>, b: &mut PolyRNS<u64>){
        debug_assert!(a.level() >= self.level(), "a.level()={} < self.level()={}", a.level(), self.level());
        debug_assert!(b.level() >= self.level(), "b.level()={} < self.level()={}", b.level(), self.level());
        self.0.iter().enumerate().for_each(|(i, ring)| ring.sub_inplace::<REDUCE>(&a.0[i], &mut b.0[i]));
    }

    #[inline(always)]
    pub fn neg<const REDUCE: REDUCEMOD>(&self, a: &PolyRNS<u64>, b: &mut PolyRNS<u64>){
        debug_assert!(a.level() >= self.level(), "a.level()={} < self.level()={}", a.level(), self.level());
        debug_assert!(b.level() >= self.level(), "b.level()={} < self.level()={}", b.level(), self.level());
        self.0.iter().enumerate().for_each(|(i, ring)| ring.neg::<REDUCE>(&a.0[i], &mut b.0[i]));
    }

    #[inline(always)]
    pub fn neg_inplace<const REDUCE: REDUCEMOD>(&self, a: &mut PolyRNS<u64>){
        debug_assert!(a.level() >= self.level(), "a.level()={} < self.level()={}", a.level(), self.level());
        self.0.iter().enumerate().for_each(|(i, ring)| ring.neg_inplace::<REDUCE>(&mut a.0[i]));
    }

    #[inline(always)]
    pub fn  mul_montgomery_external<const REDUCE:REDUCEMOD>(&self, a:&PolyRNS<Montgomery<u64>>, b:&PolyRNS<u64>, c: &mut PolyRNS<u64>){
        debug_assert!(a.level() >= self.level(), "a.level()={} < self.level()={}", a.level(), self.level());
        debug_assert!(b.level() >= self.level(), "b.level()={} < self.level()={}", b.level(), self.level());
        debug_assert!(c.level() >= self.level(), "c.level()={} < self.level()={}", c.level(), self.level());
        self.0.iter().enumerate().for_each(|(i, ring)| ring.mul_montgomery_external::<REDUCE>(&a.0[i], &b.0[i], &mut c.0[i]));
    }

    #[inline(always)]
    pub fn mul_montgomery_external_inplace<const REDUCE:REDUCEMOD>(&self, a:&PolyRNS<Montgomery<u64>>, b:&mut PolyRNS<u64>){
        debug_assert!(a.level() >= self.level(), "a.level()={} < self.level()={}", a.level(), self.level());
        debug_assert!(b.level() >= self.level(), "b.level()={} < self.level()={}", b.level(), self.level());
        self.0.iter().enumerate().for_each(|(i, ring)| ring.mul_montgomery_external_inplace::<REDUCE>(&a.0[i], &mut b.0[i]));
    }

    #[inline(always)]
    pub fn mul_scalar<const REDUCE:REDUCEMOD>(&self, a: &PolyRNS<u64>, b: &u64, c: &mut PolyRNS<u64>){
        debug_assert!(a.level() >= self.level(), "a.level()={} < self.level()={}", a.level(), self.level());
        debug_assert!(c.level() >= self.level(), "b.level()={} < self.level()={}", c.level(), self.level());
        self.0.iter().enumerate().for_each(|(i, ring)| ring.mul_scalar::<REDUCE>(&a.0[i], b, &mut c.0[i]));
    }

    #[inline(always)]
    pub fn mul_scalar_inplace<const REDUCE:REDUCEMOD>(&self, a: &u64, b: &mut PolyRNS<u64>){
        debug_assert!(b.level() >= self.level(), "b.level()={} < self.level()={}", b.level(), self.level());
        self.0.iter().enumerate().for_each(|(i, ring)| ring.mul_scalar_inplace::<REDUCE>(a, &mut b.0[i]));
    }
}