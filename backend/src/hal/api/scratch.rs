use crate::hal::layouts::{Backend, MatZnx, Module, ScalarZnx, Scratch, SvpPPol, VecZnx, VecZnxBig, VecZnxDft, VmpPMat};

pub trait ScratchOwnedAlloc<B: Backend> {
    fn alloc(size: usize) -> Self;
}

pub trait ScratchOwnedBorrow<B: Backend> {
    fn borrow(&mut self) -> &mut Scratch<B>;
}

pub trait ScratchFromBytes<B: Backend> {
    fn from_bytes(data: &mut [u8]) -> &mut Scratch<B>;
}

pub trait ScratchAvailable {
    fn available(&self) -> usize;
}

pub trait TakeSlice {
    fn take_slice<T>(&mut self, len: usize) -> (&mut [T], &mut Self);
}

pub trait TakeScalarZnx<B: Backend> {
    fn take_scalar_znx(&mut self, module: &Module<B>, cols: usize) -> (ScalarZnx<&mut [u8]>, &mut Self);
}

pub trait TakeSvpPPol<B: Backend> {
    fn take_svp_ppol(&mut self, module: &Module<B>, cols: usize) -> (SvpPPol<&mut [u8], B>, &mut Self);
}

pub trait TakeVecZnx<B: Backend> {
    fn take_vec_znx(&mut self, module: &Module<B>, cols: usize, size: usize) -> (VecZnx<&mut [u8]>, &mut Self);
}

pub trait TakeVecZnxSlice<B: Backend> {
    fn take_vec_znx_slice(
        &mut self,
        len: usize,
        module: &Module<B>,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnx<&mut [u8]>>, &mut Self);
}

pub trait TakeVecZnxBig<B: Backend> {
    fn take_vec_znx_big(&mut self, module: &Module<B>, cols: usize, size: usize) -> (VecZnxBig<&mut [u8], B>, &mut Self);
}

pub trait TakeVecZnxDft<B: Backend> {
    fn take_vec_znx_dft(&mut self, module: &Module<B>, cols: usize, size: usize) -> (VecZnxDft<&mut [u8], B>, &mut Self);
}

pub trait TakeVecZnxDftSlice<B: Backend> {
    fn take_vec_znx_dft_slice(
        &mut self,
        len: usize,
        module: &Module<B>,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnxDft<&mut [u8], B>>, &mut Self);
}

pub trait TakeVmpPMat<B: Backend> {
    fn take_vmp_pmat(
        &mut self,
        module: &Module<B>,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (VmpPMat<&mut [u8], B>, &mut Self);
}

pub trait TakeMatZnx<B: Backend> {
    fn take_mat_znx(
        &mut self,
        module: &Module<B>,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (MatZnx<&mut [u8]>, &mut Self);
}
