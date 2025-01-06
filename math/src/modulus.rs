pub mod barrett;
pub mod impl_u64;
pub mod montgomery;
pub mod prime;

pub type REDUCEMOD = u8;

pub const NONE: REDUCEMOD = 0;
pub const ONCE: REDUCEMOD = 1;
pub const TWICE: REDUCEMOD = 2;
pub const FOURTIMES: REDUCEMOD = 3;
pub const BARRETT: REDUCEMOD = 4;
pub const BARRETTLAZY: REDUCEMOD = 5;

pub trait WordOps<O> {
    fn log2(self) -> O;
    fn reverse_bits_msb(self, n: u32) -> O;
    fn mask(self) -> O;
}

impl WordOps<u64> for u64 {
    #[inline(always)]
    fn log2(self) -> u64 {
        (u64::BITS - (self - 1).leading_zeros()) as _
    }
    #[inline(always)]
    fn reverse_bits_msb(self, n: u32) -> u64 {
        self.reverse_bits() >> (usize::BITS - n)
    }
    #[inline(always)]
    fn mask(self) -> u64 {
        (1 << self.log2()) - 1
    }
}

impl WordOps<usize> for usize {
    #[inline(always)]
    fn log2(self) -> usize {
        (usize::BITS - (self - 1).leading_zeros()) as _
    }
    #[inline(always)]
    fn reverse_bits_msb(self, n: u32) -> usize {
        self.reverse_bits() >> (usize::BITS - n)
    }
    #[inline(always)]
    fn mask(self) -> usize {
        (1 << self.log2()) - 1
    }
}

pub trait ReduceOnce<O> {
    /// Assigns self-q to self if self >= q in constant time.
    /// User must ensure that 2q fits in O.
    fn reduce_once_constant_time_assign(&mut self, q: O);
    /// Returns self-q if self >= q else self in constant time.
    /// /// User must ensure that 2q fits in O.
    fn reduce_once_constant_time(&self, q: O) -> O;
    /// Assigns self-q to self if self >= q.
    /// /// User must ensure that 2q fits in O.
    fn reduce_once_assign(&mut self, q: O);
    /// Returns self-q if self >= q else self.
    /// /// User must ensure that 2q fits in O.
    fn reduce_once(&self, q: O) -> O;
}

pub trait ScalarOperations<O> {
    // Applies a parameterized modular reduction.
    fn sa_reduce_into_sa<const REDUCE: REDUCEMOD>(&self, x: &mut O);

    // Assigns a + b to c.
    fn sa_add_sb_into_sc<const REDUCE: REDUCEMOD>(&self, a: &O, b: &O, c: &mut O);

    // Assigns a + b to b.
    fn sa_add_sb_into_sb<const REDUCE: REDUCEMOD>(&self, a: &O, b: &mut O);

    // Assigns a - b to c.
    fn sa_sub_sb_into_sc<const SBRANGE: u8, const REDUCE: REDUCEMOD>(&self, a: &O, b: &O, c: &mut O);

    // Assigns a - b to b.
    fn sa_sub_sb_into_sb<const SARANGE: u8, const REDUCE: REDUCEMOD>(&self, a: &O, b: &mut O);

    // Assigns -a to a.
    fn sa_neg_into_sa<const SBRANGE: u8, const REDUCE: REDUCEMOD>(&self, a: &mut O);

    // Assigns -a to b.
    fn sa_neg_into_sb<const SARANGE: u8, const REDUCE: REDUCEMOD>(&self, a: &O, b: &mut O);

    // Assigns a * 2^64 to b.
    fn sa_prep_mont_into_sb<const REDUCE: REDUCEMOD>(
        &self,
        a: &O,
        b: &mut montgomery::Montgomery<O>,
    );

    // Assigns a * b to c.
    fn sa_mont_mul_sb_into_sc<const REDUCE: REDUCEMOD>(
        &self,
        a: &montgomery::Montgomery<O>,
        b: &O,
        c: &mut O,
    );

    // Assigns a * b to b.
    fn sa_mont_mul_sb_into_sb<const REDUCE: REDUCEMOD>(
        &self,
        a: &montgomery::Montgomery<O>,
        b: &mut O,
    );

    // Assigns a * b to c.
    fn sa_barrett_mul_sb_into_sc<const REDUCE: REDUCEMOD>(
        &self,
        a: &barrett::Barrett<O>,
        b: &O,
        c: &mut O,
    );

    // Assigns a * b to b.
    fn sa_barrett_mul_sb_into_sb<const REDUCE: REDUCEMOD>(
        &self,
        a: &barrett::Barrett<O>,
        b: &mut O,
    );

    // Assigns (a + q - b) * c to d.
    fn sa_sub_sb_mul_sc_into_sd<const VBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &O,
        b: &O,
        c: &barrett::Barrett<O>,
        d: &mut O,
    );

    // Assigns (a + q - b) * c to b.
    fn sa_sub_sb_mul_sc_into_sb<const SBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &u64,
        c: &barrett::Barrett<u64>,
        b: &mut u64,
    );

    // Assigns (a + b) * c to a.
    fn sa_add_sb_mul_sc_into_sa<const REDUCE: REDUCEMOD>(
        &self,
        b: &u64,
        c: &barrett::Barrett<u64>,
        a: &mut u64
    );

    // Assigns (a + b) * c to d.
    fn sa_add_sb_mul_sc_into_sd<const REDUCE: REDUCEMOD>(
        &self,
        a: &u64,
        b: &u64,
        c: &barrett::Barrett<u64>,
        d: &mut u64
    );
}

pub trait VectorOperations<O> {
    // Applies a parameterized modular reduction.
    fn va_reduce_into_va<const CHUNK: usize, const REDUCE: REDUCEMOD>(&self, x: &mut [O]);

    // ADD
    // vec(c) <- vec(a) + vec(b).
    fn va_add_vb_into_vc<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        va: &[O],
        vb: &[O],
        vc: &mut [O],
    );

    // vec(b) <- vec(a) + vec(b).
    fn va_add_vb_into_vb<const CHUNK: usize, const REDUCE: REDUCEMOD>(&self, a: &[O], b: &mut [O]);

    // vec(c) <- vec(a) + scalar(b).
    fn va_add_sb_into_vc<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &[O],
        b: &O,
        c: &mut [O],
    );

    // vec(b) <- vec(b) + scalar(a).
    fn va_add_sb_into_va<const CHUNK: usize, const REDUCE: REDUCEMOD>(&self, a: &O, b: &mut [O]);

    // vec(b) <- vec(a) - vec(b).
    fn va_sub_vb_into_vb<const CHUNK: usize, const VBRANGE: u8, const REDUCE: REDUCEMOD>(&self, a: &[O], b: &mut [O]);

    // vec(c) <- vec(a) - vec(b).
    fn va_sub_vb_into_vc<const CHUNK: usize, const VBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &[O],
        b: &[O],
        c: &mut [O],
    );

    // vec(a) <- -vec(a).
    fn va_neg_into_va<const CHUNK: usize, const VARANGE: u8, const REDUCE: REDUCEMOD>(&self, a: &mut [O]);

    // vec(b) <- -vec(a).
    fn va_neg_into_vb<const CHUNK: usize, const VARANGE: u8, const REDUCE: REDUCEMOD>(&self, a: &[O], b: &mut [O]);

    // vec(b) <- vec(a)
    fn va_prep_mont_into_vb<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &[O],
        b: &mut [montgomery::Montgomery<O>],
    );

    // vec(c) <- vec(a) * vec(b).
    fn va_mont_mul_vb_into_vc<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &[montgomery::Montgomery<O>],
        b: &[O],
        c: &mut [O],
    );

    // vec(b) <- vec(a) * vec(b).
    fn va_mont_mul_vb_into_vb<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &[montgomery::Montgomery<O>],
        b: &mut [O],
    );

    // vec(b) <- vec(b) * scalar(a).
    fn sa_barrett_mul_vb_into_vb<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &barrett::Barrett<u64>,
        b: &mut [u64],
    );

    // vec(c) <- vec(b) * scalar(a).
    fn sa_barrett_mul_vb_into_vc<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &barrett::Barrett<u64>,
        b: &[u64],
        c: &mut [u64],
    );

    // vec(d) <- (vec(a) + VBRANGE * q - vec(b)) * scalar(c).
    fn va_sub_vb_mul_sc_into_vd<const CHUNK: usize, const VBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &[u64],
        b: &[u64],
        c: &barrett::Barrett<u64>,
        d: &mut [u64],
    );

    // vec(b) <- (vec(a) + VBRANGE * q - vec(b)) * scalar(c).
    fn va_sub_vb_mul_sc_into_vb<const CHUNK: usize, const VBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &[u64],
        c: &barrett::Barrett<u64>,
        b: &mut [u64],
    );

    // vec(c) <- (vec(a) + scalar(b)) * scalar(c).
    fn va_add_sb_mul_sc_into_vd<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        va: &[u64],
        sb: &u64,
        sc: &barrett::Barrett<u64>,
        vd: &mut [u64],
    );

    // vec(a) <- (vec(a) + scalar(b)) * scalar(c).
    fn va_add_sb_mul_sc_into_va<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        sb: &u64,
        sc: &barrett::Barrett<u64>,
        va: &mut [u64],
    );
}
