pub mod prime;
pub mod barrett;
pub mod montgomery;

trait ReduceOnce<O>{
    fn reduce_once_assign(&mut self, q: O);
    fn reduce_once(&self, q:O) -> O;
}

impl ReduceOnce<u64> for u64{
    #[inline(always)]
    fn reduce_once_assign(&mut self, q: u64){
        if *self >= q{
            *self -= q
        }
    }
    #[inline(always)]
    fn reduce_once(&self, q:u64) -> u64{
        if *self >= q {
            *self - q
        } else {
            *self
        }
    }
}
