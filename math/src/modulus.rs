pub mod prime;
pub mod barrett;
pub mod montgomery;
pub mod impl_u64;

pub type REDUCEMOD = u8;

pub const NONE: REDUCEMOD = 0;
pub const ONCE: REDUCEMOD = 1;
pub const TWICE: REDUCEMOD = 2;
pub const FOURTIMES: REDUCEMOD = 3;
pub const BARRETT: REDUCEMOD = 4;
pub const BARRETTLAZY: REDUCEMOD = 5;

pub trait WordOps<O>{
    fn log2(self) -> O;
    fn reverse_bits_msb(self, n:u32) -> O;
    fn mask(self) -> O;
}

impl WordOps<u64> for u64{
    #[inline(always)]
    fn log2(self) -> u64{
        (u64::BITS - (self-1).leading_zeros()) as _
    }
    #[inline(always)]
    fn reverse_bits_msb(self, n: u32) -> u64{
        self.reverse_bits() >> (usize::BITS - n)
    }
    #[inline(always)]
    fn mask(self) -> u64{
        (1<<self.log2())-1
    }
}

impl WordOps<usize> for usize{
    #[inline(always)]
    fn log2(self) -> usize{
        (usize::BITS - (self-1).leading_zeros()) as _
    }
    #[inline(always)]
    fn reverse_bits_msb(self, n: u32) -> usize{
        self.reverse_bits() >> (usize::BITS - n)
    }
    #[inline(always)]
    fn mask(self) -> usize{
        (1<<self.log2())-1
    }
}

pub trait ReduceOnce<O>{
    /// Assigns self-q to self if self >= q in constant time.
    /// User must ensure that 2q fits in O.
    fn reduce_once_constant_time_assign(&mut self, q: O);
    /// Returns self-q if self >= q else self in constant time.
    /// /// User must ensure that 2q fits in O.
    fn reduce_once_constant_time(&self, q:O) -> O;
    /// Assigns self-q to self if self >= q.
    /// /// User must ensure that 2q fits in O.
    fn reduce_once_assign(&mut self, q: O);
    /// Returns self-q if self >= q else self.
    /// /// User must ensure that 2q fits in O.
    fn reduce_once(&self, q:O) -> O;
}

pub trait ScalarOperations<O>{

    // Applies a parameterized modular reduction.
    fn sa_reduce_into_sa<const REDUCE:REDUCEMOD>(&self, x: &mut O);

    // Assigns a + b to c.
    fn sa_add_sb_into_sc<const REDUCE:REDUCEMOD>(&self, a: &O, b:&O, c: &mut O);

    // Assigns a + b to b.
    fn sa_add_sb_into_sb<const REDUCE:REDUCEMOD>(&self, a: &O, b: &mut O);

    // Assigns a - b to c.
    fn sa_sub_sb_into_sc<const REDUCE:REDUCEMOD>(&self, a: &O, b:&O, c: &mut O);

    // Assigns b - a to b.
    fn sa_sub_sb_into_sb<const REDUCE:REDUCEMOD>(&self, a: &O, b: &mut O);

    // Assigns -a to a.
    fn sa_neg_into_sa<const REDUCE:REDUCEMOD>(&self, a:&mut O);

    // Assigns -a to b.
    fn sa_neg_into_sb<const REDUCE:REDUCEMOD>(&self, a: &O, b:&mut O);

    // Assigns a * 2^64 to b.
    fn sa_prep_mont_into_sb<const REDUCE:REDUCEMOD>(&self, a: &O, b: &mut montgomery::Montgomery<O>);

    // Assigns a * b to c.
    fn sa_mont_mul_sb_into_sc<const REDUCE:REDUCEMOD>(&self, a:&montgomery::Montgomery<O>, b:&O, c: &mut O);

    // Assigns a * b to b.
    fn sa_mont_mul_sb_into_sb<const REDUCE:REDUCEMOD>(&self, a:&montgomery::Montgomery<O>, b:&mut O);

    // Assigns a * b to c.
    fn sa_barrett_mul_sb_into_sc<const REDUCE:REDUCEMOD>(&self, a: &barrett::Barrett<O>, b:&O, c: &mut O);

    // Assigns a * b to b.
    fn sa_barrett_mul_sb_into_sb<const REDUCE:REDUCEMOD>(&self, a:&barrett::Barrett<O>, b:&mut O);

    // Assigns (a + 2q - b) * c to d.
    fn sa_sub_sb_mul_sc_into_sd<const REDUCE:REDUCEMOD>(&self, a: &O, b: &O, c: &barrett::Barrett<O>, d: &mut O);

    // Assigns (a + 2q - b) * c to b.
    fn sa_sub_sb_mul_sc_into_sb<const REDUCE:REDUCEMOD>(&self, a: &u64, c: &barrett::Barrett<u64>, b: &mut u64);
}

pub trait VectorOperations<O>{

    // Applies a parameterized modular reduction.
    fn va_reduce_into_va<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, x: &mut [O]);

    // ADD
    // Assigns a[i] + b[i] to c[i]
    fn va_add_vb_into_vc<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[O], b:&[O], c: &mut [O]);

    // Assigns a[i] + b[i] to b[i]
    fn va_add_vb_into_vb<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[O], b: &mut [O]);

    // Assigns a[i] + b to c[i]
    fn va_add_sb_into_vc<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[O], b:&O, c:&mut [O]);

    // Assigns b[i] + a to b[i]
    fn sa_add_vb_into_vb<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a:&O, b:&mut [O]);

    // SUB
    // Assigns a[i] - b[i] to b[i]
    fn va_sub_vb_into_vb<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[O], b: &mut [O]);

    // Assigns a[i] - b[i] to c[i]
    fn va_sub_vb_into_vc<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[O], b:&[O], c: &mut [O]);

    // NEG
    // Assigns -a[i] to a[i].
    fn va_neg_into_va<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &mut [O]);

    // Assigns -a[i] to a[i].
    fn va_neg_into_vb<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[O], b: &mut [O]);

    // MUL MONTGOMERY
    // Assigns a * 2^64 to b.
    fn va_prep_mont_into_vb<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[O], b: &mut [montgomery::Montgomery<O>]);

    // Assigns a[i] * b[i] to c[i].
    fn va_mont_mul_vb_into_vc<const CHUNK:usize,const REDUCE:REDUCEMOD>(&self, a:&[montgomery::Montgomery<O>], b:&[O], c: &mut [O]);

    // Assigns a[i] * b[i] to b[i].
    fn va_mont_mul_vb_into_vb<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a:&[montgomery::Montgomery<O>], b:&mut [O]);

    // MUL BARRETT
    // Assigns a * b[i] to b[i].
    fn sa_barrett_mul_vb_into_vb<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a:& barrett::Barrett<u64>, b:&mut [u64]);

    // Assigns a * b[i] to c[i].
    fn sa_barrett_mul_vb_into_vc<const CHUNK:usize,const REDUCE:REDUCEMOD>(&self, a:& barrett::Barrett<u64>, b:&[u64], c: &mut [u64]);

    // OTHERS
    // Assigns (a[i] + 2q - b[i]) * c to d[i].
    fn va_sub_vb_mul_sc_into_vd<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[u64], b: &[u64], c: &barrett::Barrett<u64>, d: &mut [u64]);

    // Assigns (a[i] + 2q - b[i]) * c to b[i].
    fn va_sub_vb_mul_sc_into_vb<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[u64], c: &barrett::Barrett<u64>, b: &mut [u64]);
}




