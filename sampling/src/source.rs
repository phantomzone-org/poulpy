use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_core::{OsRng, RngCore};

const MAXF64: f64 = 9007199254740992.0;

pub struct Source {
    source: ChaCha8Rng,
}

pub fn new_seed() -> [u8; 32] {
    let mut seed = [0u8; 32];
    OsRng.fill_bytes(&mut seed);
    seed
}

impl Source {
    pub fn new(seed: [u8; 32]) -> Source {
        Source {
            source: ChaCha8Rng::from_seed(seed),
        }
    }

    pub fn new_seed(&mut self) -> [u8; 32] {
        let mut seed: [u8; 32] = [0u8; 32];
        self.source.fill_bytes(&mut seed);
        seed
    }

    pub fn branch(&mut self) -> Self {
        Source::new(self.new_seed())
    }

    #[inline(always)]
    pub fn next_u64n(&mut self, max: u64, mask: u64) -> u64 {
        let mut x: u64 = self.next_u64() & mask;
        while x >= max {
            x = self.next_u64() & mask;
        }
        x
    }

    #[inline(always)]
    pub fn next_f64(&mut self, min: f64, max: f64) -> f64 {
        min + ((self.next_u64() << 11 >> 11) as f64) / MAXF64 * (max - min)
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

    #[inline(always)]
    fn try_fill_bytes(&mut self, bytes: &mut [u8]) -> Result<(), rand_core::Error> {
        self.source.try_fill_bytes(bytes)
    }
}
