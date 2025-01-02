pub mod prime;
pub mod barrett;
pub mod montgomery;
pub mod operations;

use crate::modulus::ReduceOnce;

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