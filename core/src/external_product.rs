use base2k::{FFT64, Module, Scratch};

pub trait ExternalProductScratchSpace {
    fn external_product_scratch_space(module: &Module<FFT64>, res_size: usize, lhs: usize, rhs: usize) -> usize;
}

pub trait ExternalProduct<DataLhs, DataRhs> {
    type Lhs;
    type Rhs;
    fn external_product(&mut self, module: &Module<FFT64>, lhs: &Self::Lhs, rhs: &Self::Rhs, scratch: &mut Scratch);
}
pub trait ExternalProductInplaceScratchSpace {
    fn external_product_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, rhs: usize) -> usize;
}

pub trait ExternalProductInplace<DataRhs> {
    type Rhs;
    fn external_product_inplace(&mut self, module: &Module<FFT64>, rhs: &Self::Rhs, scratch: &mut Scratch);
}
