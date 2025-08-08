use crate::hal::layouts::{Backend, MatZnx, Module, ScalarZnx, Scratch, SvpPPol, VecZnx, VecZnxBig, VecZnxDft, VmpPMat};

/// Allocates a new [crate::hal::layouts::ScratchOwned] of `size` aligned bytes.
pub trait ScratchOwnedAlloc<B: Backend> {
    fn alloc(size: usize) -> Self;
}

/// Borrows a slice of bytes into a [Scratch].
pub trait ScratchOwnedBorrow<B: Backend> {
    fn borrow(&mut self) -> &mut Scratch<B>;
}

/// Wrap an array of mutable borrowed bytes into a [Scratch].
pub trait ScratchFromBytes<B: Backend> {
    fn from_bytes(data: &mut [u8]) -> &mut Scratch<B>;
}

/// Returns how many bytes left can be taken from the scratch.
pub trait ScratchAvailable {
    fn available(&self) -> usize;
}

/// Takes a slice of bytes from a [Scratch] and return a new [Scratch] minus the taken array of bytes.
pub trait TakeSlice {
    fn take_slice<T>(&mut self, len: usize) -> (&mut [T], &mut Self);
}

/// Take a slice of bytes from a [Scratch], wraps it into a [ScalarZnx] and returns it
/// as well as a new [Scratch] minus the taken array of bytes.
pub trait TakeScalarZnx<B: Backend> {
    fn take_scalar_znx(&mut self, module: &Module<B>, cols: usize) -> (ScalarZnx<&mut [u8]>, &mut Self);
}

/// Take a slice of bytes from a [Scratch], wraps it into a [SvpPPol] and returns it
/// as well as a new [Scratch] minus the taken array of bytes.
pub trait TakeSvpPPol<B: Backend> {
    fn take_svp_ppol(&mut self, module: &Module<B>, cols: usize) -> (SvpPPol<&mut [u8], B>, &mut Self);
}

/// Take a slice of bytes from a [Scratch], wraps it into a [VecZnx] and returns it
/// as well as a new [Scratch] minus the taken array of bytes.
pub trait TakeVecZnx<B: Backend> {
    fn take_vec_znx(&mut self, module: &Module<B>, cols: usize, size: usize) -> (VecZnx<&mut [u8]>, &mut Self);
}

/// Take a slice of bytes from a [Scratch], slices it into a vector of [VecZnx] aand returns it
/// as well as a new [Scratch] minus the taken array of bytes.
pub trait TakeVecZnxSlice<B: Backend> {
    fn take_vec_znx_slice(
        &mut self,
        len: usize,
        module: &Module<B>,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnx<&mut [u8]>>, &mut Self);
}

/// Take a slice of bytes from a [Scratch], wraps it into a [VecZnxBig] and returns it
/// as well as a new [Scratch] minus the taken array of bytes.
pub trait TakeVecZnxBig<B: Backend> {
    fn take_vec_znx_big(&mut self, module: &Module<B>, cols: usize, size: usize) -> (VecZnxBig<&mut [u8], B>, &mut Self);
}

/// Take a slice of bytes from a [Scratch], wraps it into a [VecZnxDft] and returns it
/// as well as a new [Scratch] minus the taken array of bytes.
pub trait TakeVecZnxDft<B: Backend> {
    fn take_vec_znx_dft(&mut self, module: &Module<B>, cols: usize, size: usize) -> (VecZnxDft<&mut [u8], B>, &mut Self);
}

/// Take a slice of bytes from a [Scratch], slices it into a vector of [VecZnxDft] and returns it
/// as well as a new [Scratch] minus the taken array of bytes.
pub trait TakeVecZnxDftSlice<B: Backend> {
    fn take_vec_znx_dft_slice(
        &mut self,
        len: usize,
        module: &Module<B>,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnxDft<&mut [u8], B>>, &mut Self);
}

/// Take a slice of bytes from a [Scratch], wraps it into a [VmpPMat] and returns it
/// as well as a new [Scratch] minus the taken array of bytes.
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

/// Take a slice of bytes from a [Scratch], wraps it into a [MatZnx] and returns it
/// as well as a new [Scratch] minus the taken array of bytes.
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

/// Take a slice of bytes from a [Scratch], wraps it into the template's type and returns it
/// as well as a new [Scratch] minus the taken array of bytes.
pub trait TakeLike<'a, B: Backend, T> {
    type Output;
    fn take_like(&'a mut self, template: &T) -> (Self::Output, &'a mut Self);
}
