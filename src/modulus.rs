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

pub trait WordOperations<O>{

    // Applies a parameterized modular reduction.
    fn word_reduce_assign<const REDUCE:REDUCEMOD>(&self, x: &mut O);

    // Assigns a + b to c.
    fn word_add_binary_assign<const REDUCE:REDUCEMOD>(&self, a: &O, b:&O, c: &mut O);

    // Assigns a + b to b.
    fn word_add_unary_assign<const REDUCE:REDUCEMOD>(&self, a: &O, b: &mut O);

    // Assigns a - b to c.
    fn word_sub_binary_assign<const REDUCE:REDUCEMOD>(&self, a: &O, b:&O, c: &mut O);

    // Assigns b - a to b.
    fn word_sub_unary_assign<const REDUCE:REDUCEMOD>(&self, a: &O, b: &mut O);

    // Assigns -a to a.
    fn word_neg_assign<const REDUCE:REDUCEMOD>(&self, a:&mut O);

    // Assigns a * 2^64 to b.
    fn word_prepare_montgomery_assign<const REDUCE:REDUCEMOD>(&self, a: &O, b: &mut montgomery::Montgomery<O>);

    // Assigns a * b to c.
    fn word_mul_montgomery_external_binary_assign<const REDUCE:REDUCEMOD>(&self, a:&montgomery::Montgomery<O>, b:&O, c: &mut O);

    // Assigns a * b to b.
    fn word_mul_montgomery_external_unary_assign<const REDUCE:REDUCEMOD>(&self, a:&montgomery::Montgomery<O>, b:&mut O);
}

pub trait VecOperations<O>{

    // Applies a parameterized modular reduction.
    fn vec_reduce_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, x: &mut [O]);

    // Assigns a[i] + b[i] to c[i]
    fn vec_add_binary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[O], b:&[O], c: &mut [O]);

    // Assigns a[i] + b[i] to b[i]
    fn vec_add_unary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[O], b: &mut [O]);

    // Assigns a[i] - b[i] to c[i]
    fn vec_sub_binary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[O], b:&[O], c: &mut [O]);

    // Assigns a[i] - b[i] to b[i]
    fn vec_sub_unary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[O], b: &mut [O]);

    // Assigns -a[i] to a[i].
    fn vec_neg_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &mut [O]);

    // Assigns a * 2^64 to b.
    fn vec_prepare_montgomery_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[O], b: &mut [montgomery::Montgomery<O>]);

    // Assigns a[i] * b[i] to c[i].
    fn vec_mul_montgomery_external_binary_assign<const CHUNK:usize,const REDUCE:REDUCEMOD>(&self, a:&[montgomery::Montgomery<O>], b:&[O], c: &mut [O]);

    // Assigns a[i] * b[i] to b[i].
    fn vec_mul_montgomery_external_unary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a:&[montgomery::Montgomery<O>], b:&mut [O]);
}




