use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use rand_core::RngCore;

const MAXF64: f64 = 9007199254740992.0;

pub struct Source {
    source: ChaCha8Rng,
}

impl Source {
    pub fn new(seed: [u8; 32]) -> Source {
        Source {
            source: ChaCha8Rng::from_seed(seed),
        }
    }

    pub fn branch(&mut self) -> ([u8; 32], Self) {
        let seed: [u8; 32] = self.new_seed();
        (seed, Source::new(seed))
    }

    pub fn new_seed(&mut self) -> [u8; 32] {
        let mut seed: [u8; 32] = [0u8; 32];
        self.fill_bytes(&mut seed);
        seed
    }

    #[inline(always)]
    pub fn next_u64n(&mut self, max: u64, mask: u64) -> u64 {
        let mut x: u64 = self.next_u64() & mask;
        while x >= max {
            x = self.next_u64() & mask;
        }
        x
    }

    /// Returns a uniformly distributed f64 in [min, max).
    ///
    /// # Panics
    /// Panics if `min > max`.
    #[inline(always)]
    pub fn next_f64(&mut self, min: f64, max: f64) -> f64 {
        debug_assert!(min <= max, "next_f64: min ({min}) > max ({max})");
        min + ((self.next_u64() << 11 >> 11) as f64) / MAXF64 * (max - min)
    }

    #[inline(always)]
    pub fn next_i32(&mut self) -> i32 {
        self.next_u32() as i32
    }

    #[inline(always)]
    pub fn next_i64(&mut self) -> i64 {
        self.next_u64() as i64
    }

    #[inline(always)]
    pub fn next_i128(&mut self) -> i128 {
        self.next_u128() as i128
    }

    #[inline(always)]
    pub fn next_u128(&mut self) -> u128 {
        (self.next_u64() as u128) << 64 | (self.next_u64() as u128)
    }
}

impl RngCore for Source {
    #[inline(always)]
    fn next_u32(&mut self) -> u32 {
        self.source.next_u32()
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        self.source.next_u64()
    }

    #[inline(always)]
    fn fill_bytes(&mut self, bytes: &mut [u8]) {
        self.source.fill_bytes(bytes)
    }
}
