pub mod prime;
pub mod barrett;
pub mod montgomery;
pub mod shoup;
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

impl ReduceOnce<u64> for u64{
    #[inline(always)]
    fn reduce_once_constant_time_assign(&mut self, q: u64){
        debug_assert!(q < 0x8000000000000000, "2q >= 2^64");
        *self -= (q.wrapping_sub(*self)>>63)*q;
    }
    
    #[inline(always)]
    fn reduce_once_constant_time(&self, q:u64) -> u64{
        debug_assert!(q < 0x8000000000000000, "2q >= 2^64");
        self - (q.wrapping_sub(*self)>>63)*q
    }

    #[inline(always)]
    fn reduce_once_assign(&mut self, q: u64){
        debug_assert!(q < 0x8000000000000000, "2q >= 2^64");
        *self = *self.min(&mut self.wrapping_sub(q))
    }

    #[inline(always)]
    fn reduce_once(&self, q:u64) -> u64{
        debug_assert!(q < 0x8000000000000000, "2q >= 2^64");
        *self.min(&mut self.wrapping_sub(q))
    }
}


pub trait Operations<O>{
    // Assigns a + b to c.
    fn add_binary_assign<const REDUCE:REDUCEMOD>(&self, a: &O, b:&O, c: &mut O);

    // Assigns a + b to b.
    fn add_unary_assign<const REDUCE:REDUCEMOD>(&self, a: &O, b: &mut O);

    // Assigns a[i] + b[i] to c[i]
    fn add_vec_binary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[O], b:&[O], c: &mut [O]);

    // Assigns a[i] + b[i] to b[i]
    fn add_vec_unary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[O], b: &mut [O]);

    // Assigns a - b to c.
    fn sub_binary_assign<const REDUCE:REDUCEMOD>(&self, a: &O, b:&O, c: &mut O);

    // Assigns b - a to b.
    fn sub_unary_assign<const REDUCE:REDUCEMOD>(&self, a: &O, b: &mut O);

    // Assigns a[i] - b[i] to c[i]
    fn sub_vec_binary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[O], b:&[O], c: &mut [O]);

    // Assigns a[i] - b[i] to b[i]
    fn sub_vec_unary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[O], b: &mut [O]);

    // Assigns -a to a.
    fn neg_assign<const REDUCE:REDUCEMOD>(&self, a:&mut O);

    // Assigns -a[i] to a[i].
    fn neg_vec_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &mut [O]);

    // Assigns a * b to c.
    fn mul_montgomery_external_binary_assign<const REDUCE:REDUCEMOD>(&self, a:&montgomery::Montgomery<O>, b:&O, c: &mut O);

    // Assigns a * b to b.
    fn mul_montgomery_external_unary_assign<const REDUCE:REDUCEMOD>(&self, a:&montgomery::Montgomery<O>, b:&mut O);

    // Assigns a[i] * b[i] to c[i].
    fn mul_vec_montgomery_external_binary_assign<const CHUNK:usize,const REDUCE:REDUCEMOD>(&self, a:&[montgomery::Montgomery<O>], b:&[O], c: &mut [O]);

    // Assigns a[i] * b[i] to b[i].
    fn mul_vec_montgomery_external_unary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a:&[montgomery::Montgomery<O>], b:&mut [O]);
}




