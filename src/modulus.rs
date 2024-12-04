pub(crate) mod prime;
pub(crate) mod montgomery;
pub(crate) mod barrett;

trait ReduceOnce<O>{
    fn reduce_once_assign(&mut self, q: O);
    fn reduce_once(&self, q:O) -> O;
}

impl ReduceOnce<u64> for u64{
    fn reduce_once_assign(&mut self, q: u64){
        if *self >= q{
            *self -= q
        }
    }

    fn reduce_once(&self, q:u64) -> u64{
        if *self >= q {
            *self - q
        } else {
            *self
        }
    }
}