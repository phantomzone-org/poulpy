pub mod ntt;

pub trait DFT<O> {
    fn forward_inplace_lazy(&self, x: &mut [O]);
    fn forward_inplace(&self, x: &mut [O]);
    fn backward_inplace_lazy(&self, x: &mut [O]);
    fn backward_inplace(&self, x: &mut [O]);
}
