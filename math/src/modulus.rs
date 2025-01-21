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
    fn log2(self) -> usize;
    fn reverse_bits_msb(self, n: u32) -> O;
    fn mask(self) -> O;
}

impl WordOps<u64> for u64 {
    #[inline(always)]
    fn log2(self) -> usize {
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
    fn sa_sub_sb_into_sc<const SBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &O,
        b: &O,
        c: &mut O,
    );

    // Assigns a - b to b.
    fn sa_sub_sb_into_sb<const SARANGE: u8, const REDUCE: REDUCEMOD>(&self, a: &O, b: &mut O);

    // Assigns a - b to a.
    fn sa_sub_sb_into_sa<const SARANGE: u8, const REDUCE: REDUCEMOD>(&self, b: &O, a: &mut O);

    // Assigns -a to a.
    fn sa_neg_into_sa<const SBRANGE: u8, const REDUCE: REDUCEMOD>(&self, a: &mut O);

    // Assigns -a to b.
    fn sa_neg_into_sb<const SARANGE: u8, const REDUCE: REDUCEMOD>(&self, a: &O, b: &mut O);

    // Assigns a * 2^64 to a.
    fn sa_prepare_montgomery_into_sa<const REDUCE: REDUCEMOD>(
        &self,
        a: &mut montgomery::Montgomery<O>,
    );

    // Assigns a * 2^64 to b.
    fn sa_prepare_montgomery_into_sb<const REDUCE: REDUCEMOD>(
        &self,
        a: &O,
        b: &mut montgomery::Montgomery<O>,
    );

    // Assigns a * b to c.
    fn sa_mul_sb_montgomery_into_sc<const REDUCE: REDUCEMOD>(
        &self,
        a: &O,
        b: &montgomery::Montgomery<O>,
        c: &mut O,
    );

    // Assigns a * b + c to c.
    fn sa_mul_sb_montgomery_add_sc_into_sc<const REDUCE1: REDUCEMOD, const REDUCE2: REDUCEMOD>(
        &self,
        a: &O,
        b: &montgomery::Montgomery<O>,
        c: &mut O,
    );

    // Assigns a * b to b.
    fn sa_mul_sb_montgomery_into_sa<const REDUCE: REDUCEMOD>(
        &self,
        b: &montgomery::Montgomery<O>,
        a: &mut O,
    );

    // Assigns a * b to c.
    fn sa_mul_sb_barrett_into_sc<const REDUCE: REDUCEMOD>(
        &self,
        a: &O,
        b: &barrett::Barrett<O>,
        c: &mut O,
    );

    // Assigns a * b to a.
    fn sa_mul_sb_barrett_into_sa<const REDUCE: REDUCEMOD>(
        &self,
        b: &barrett::Barrett<O>,
        a: &mut O,
    );

    // Assigns (a + q - b) * c to d.
    fn sa_sub_sb_mul_sc_barrett_into_sd<const VBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &O,
        b: &O,
        c: &barrett::Barrett<O>,
        d: &mut O,
    );

    // Assigns (a + q - b) * c to b.
    fn sa_sub_sb_mul_sc_barrett_into_sb<const SBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &u64,
        c: &barrett::Barrett<u64>,
        b: &mut u64,
    );

    // Assigns (a + b) * c to a.
    fn sa_add_sb_mul_sc_barrett_into_sa<const REDUCE: REDUCEMOD>(
        &self,
        b: &u64,
        c: &barrett::Barrett<u64>,
        a: &mut u64,
    );

    // Assigns (a + b) * c to d.
    fn sa_add_sb_mul_sc_barrett_into_sd<const REDUCE: REDUCEMOD>(
        &self,
        a: &u64,
        b: &u64,
        c: &barrett::Barrett<u64>,
        d: &mut u64,
    );

    // Assigns (a - b + c) * d to e.
    fn sb_sub_sa_add_sc_mul_sd_barrett_into_se<const SBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &u64,
        b: &u64,
        c: &u64,
        d: &barrett::Barrett<u64>,
        e: &mut u64,
    );

    fn sb_sub_sa_add_sc_mul_sd_barrett_into_sa<const SBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        b: &u64,
        c: &u64,
        d: &barrett::Barrett<u64>,
        a: &mut u64,
    );

    fn sa_rsh_sb_mask_sc_into_sa(&self, c: &usize, b: &u64, a: &mut u64);

    fn sa_rsh_sb_mask_sc_into_sd(&self, a: &u64, b: &usize, c: &u64, d: &mut u64);

    fn sa_rsh_sb_mask_sc_add_sd_into_sd(&self, a: &u64, b: &usize, c: &u64, d: &mut u64);

    fn sa_signed_digit_into_sb<const CARRYOVERWRITE: bool, const BALANCED: bool>(
        &self,
        a: &u64,
        base: &u64,
        shift: &usize,
        mask: &u64,
        carry: &mut u64,
        b: &mut u64,
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
    fn va_sub_vb_into_vb<const CHUNK: usize, const VBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &[O],
        b: &mut [O],
    );

    // vec(a) <- vec(a) - vec(b).
    fn va_sub_vb_into_va<const CHUNK: usize, const VBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        b: &[O],
        a: &mut [O],
    );

    // vec(c) <- vec(a) - vec(b).
    fn va_sub_vb_into_vc<const CHUNK: usize, const VBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &[O],
        b: &[O],
        c: &mut [O],
    );

    // vec(a) <- -vec(a).
    fn va_neg_into_va<const CHUNK: usize, const VARANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &mut [O],
    );

    // vec(b) <- -vec(a).
    fn va_neg_into_vb<const CHUNK: usize, const VARANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &[O],
        b: &mut [O],
    );

    // vec(b) <- vec(a)
    fn va_prep_mont_into_vb<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &[O],
        b: &mut [montgomery::Montgomery<O>],
    );

    // vec(a) <- vec(a).
    fn va_prepare_montgomery_into_va<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &mut [montgomery::Montgomery<O>],
    );

    // vec(c) <- vec(a) * vec(b).
    fn va_mul_vb_montgomery_into_vc<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &[O],
        b: &[montgomery::Montgomery<O>],
        c: &mut [O],
    );

    // vec(c) <- vec(a) * vec(b) + vec(c).
    fn va_mul_vb_montgomery_add_vc_into_vc<
        const CHUNK: usize,
        const REDUCE1: REDUCEMOD,
        const REDUCE2: REDUCEMOD,
    >(
        &self,
        a: &[O],
        b: &[montgomery::Montgomery<O>],
        c: &mut [O],
    );

    // vec(a) <- vec(a) * vec(b).
    fn va_mul_vb_montgomery_into_va<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        b: &[montgomery::Montgomery<O>],
        a: &mut [O],
    );

    // vec(b) <- vec(a) * scalar(b).
    fn va_mul_sb_barrett_into_va<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        b: &barrett::Barrett<u64>,
        a: &mut [u64],
    );

    // vec(c) <- vec(a) * scalar(b).
    fn va_mul_sb_barrett_into_vc<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &[u64],
        b: &barrett::Barrett<u64>,
        c: &mut [u64],
    );

    // vec(d) <- (vec(a) + VBRANGE * q - vec(b)) * scalar(c).
    fn va_sub_vb_mul_sc_barrett_into_vd<
        const CHUNK: usize,
        const VBRANGE: u8,
        const REDUCE: REDUCEMOD,
    >(
        &self,
        a: &[u64],
        b: &[u64],
        c: &barrett::Barrett<u64>,
        d: &mut [u64],
    );

    // vec(b) <- (vec(a) + VBRANGE * q - vec(b)) * scalar(c).
    fn va_sub_vb_mul_sc_barrett_into_vb<
        const CHUNK: usize,
        const VBRANGE: u8,
        const REDUCE: REDUCEMOD,
    >(
        &self,
        a: &[u64],
        c: &barrett::Barrett<u64>,
        b: &mut [u64],
    );

    // vec(c) <- (vec(a) + scalar(b)) * scalar(c).
    fn va_add_sb_mul_sc_barrett_into_vd<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        va: &[u64],
        sb: &u64,
        sc: &barrett::Barrett<u64>,
        vd: &mut [u64],
    );

    // vec(a) <- (vec(a) + scalar(b)) * scalar(c).
    fn va_add_sb_mul_sc_barrett_into_va<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        sb: &u64,
        sc: &barrett::Barrett<u64>,
        va: &mut [u64],
    );

    // vec(e) <- (vec(b) - vec(a) + scalar(c)) * scalar(e).
    fn vb_sub_va_add_sc_mul_sd_barrett_into_ve<
        const CHUNK: usize,
        const VBRANGE: u8,
        const REDUCE: REDUCEMOD,
    >(
        &self,
        va: &[u64],
        vb: &[u64],
        sc: &u64,
        sd: &barrett::Barrett<u64>,
        ve: &mut [u64],
    );

    // vec(a) <- (vec(b) - vec(a) + scalar(c)) * scalar(e).
    fn vb_sub_va_add_sc_mul_sd_barrett_into_va<
        const CHUNK: usize,
        const VBRANGE: u8,
        const REDUCE: REDUCEMOD,
    >(
        &self,
        vb: &[u64],
        sc: &u64,
        sd: &barrett::Barrett<u64>,
        va: &mut [u64],
    );

    // vec(a) <- (vec(a)>>scalar(b)) & scalar(c).
    fn va_rsh_sb_mask_sc_into_va<const CHUNK: usize>(&self, sb: &usize, sc: &u64, va: &mut [u64]);

    // vec(d) <- (vec(a)>>scalar(b)) & scalar(c).
    fn va_rsh_sb_mask_sc_into_vd<const CHUNK: usize>(
        &self,
        va: &[u64],
        sb: &usize,
        sc: &u64,
        vd: &mut [u64],
    );

    // vec(d) <- vec(d) + (vec(a)>>scalar(b)) & scalar(c).
    fn va_rsh_sb_mask_sc_add_vd_into_vd<const CHUNK: usize>(
        &self,
        va: &[u64],
        sb: &usize,
        sc: &u64,
        vd: &mut [u64],
    );

    // vec(c) <- i-th unsigned digit base 2^{sb} of vec(a).
    // vec(c) is ensured to be in the range [0, 2^{sb}-1[ with E[vec(c)] = 2^{sb}-1.
    fn va_ith_digit_unsigned_base_sb_into_vc<const CHUNK: usize>(
        &self,
        i: usize,
        va: &[u64],
        sb: &usize,
        vc: &mut [u64],
    );

    // vec(c) <- i-th signed digit base 2^{w} of vec(a).
    // Reads the carry of the i-1-th iteration and write the carry on the i-th iteration on carry.
    // if i > 0, carry of the i-1th iteration must be provided.
    // if BALANCED: vec(c) is ensured to be [-2^{sb-1}, 2^{sb-1}[ with E[vec(c)] = 0, else E[vec(c)] = -0.5
    fn va_ith_digit_signed_base_sb_into_vc<const CHUNK: usize, const BALANCED: bool>(
        &self,
        i: usize,
        va: &[u64],
        sb: &usize,
        carry: &mut [u64],
        vc: &mut [u64],
    );
}
