use crate::{
    api::{ModuleN, SvpPPolBytesOf, VecZnxBigBytesOf, VecZnxDftBytesOf, VmpPMatBytesOf},
    layouts::{Backend, MatZnx, ScalarZnx, Scratch, SvpPPol, VecZnx, VecZnxBig, VecZnxDft, VmpPMat},
};

/// Allocates a new [crate::layouts::ScratchOwned] of `size` aligned bytes.
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

impl<B: Backend> ScratchTakeBasic for Scratch<B> where Self: TakeSlice {}

pub trait ScratchTakeBasic
where
    Self: TakeSlice,
{
    fn take_scalar_znx(&mut self, n: usize, cols: usize) -> (ScalarZnx<&mut [u8]>, &mut Self)
    {
        let (take_slice, rem_slice) = self.take_slice(ScalarZnx::bytes_of(n, cols));
        (
            ScalarZnx::from_data(take_slice, n, cols),
            rem_slice,
        )
    }

    fn take_svp_ppol<M, B: Backend>(&mut self, module: &M, cols: usize) -> (SvpPPol<&mut [u8], B>, &mut Self)
    where
        M: SvpPPolBytesOf + ModuleN,
    {
        let (take_slice, rem_slice) = self.take_slice(module.bytes_of_svp_ppol(cols));
        (SvpPPol::from_data(take_slice, module.n(), cols), rem_slice)
    }

    fn take_vec_znx(&mut self, n: usize, cols: usize, size: usize) -> (VecZnx<&mut [u8]>, &mut Self){
        let (take_slice, rem_slice) = self.take_slice(VecZnx::bytes_of(n, cols, size));
        (
            VecZnx::from_data(take_slice, n, cols, size),
            rem_slice,
        )
    }

    fn take_vec_znx_big<M, B: Backend>(&mut self, module: &M, cols: usize, size: usize) -> (VecZnxBig<&mut [u8], B>, &mut Self)
    where
        M: VecZnxBigBytesOf + ModuleN,
    {
        let (take_slice, rem_slice) = self.take_slice(module.bytes_of_vec_znx_big(cols, size));
        (
            VecZnxBig::from_data(take_slice, module.n(), cols, size),
            rem_slice,
        )
    }

    fn take_vec_znx_dft<M, B: Backend>(&mut self, module: &M, cols: usize, size: usize) -> (VecZnxDft<&mut [u8], B>, &mut Self)
    where
        M: VecZnxDftBytesOf + ModuleN,
    {
        let (take_slice, rem_slice) = self.take_slice(module.bytes_of_vec_znx_dft(cols, size));

        (
            VecZnxDft::from_data(take_slice, module.n(), cols, size),
            rem_slice,
        )
    }

    fn take_vec_znx_dft_slice<M, B: Backend>(
        &mut self,
        module: &M,
        len: usize,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnxDft<&mut [u8], B>>, &mut Self)
    where
        M: VecZnxDftBytesOf + ModuleN,
    {
        let mut scratch: &mut Self = self;
        let mut slice: Vec<VecZnxDft<&mut [u8], B>> = Vec::with_capacity(len);
        for _ in 0..len {
            let (znx, new_scratch) = scratch.take_vec_znx_dft(module, cols, size);
            scratch = new_scratch;
            slice.push(znx);
        }
        (slice, scratch)
    }

    fn take_vec_znx_slice(&mut self, n: usize, len: usize, cols: usize, size: usize) -> (Vec<VecZnx<&mut [u8]>>, &mut Self){
        let mut scratch: &mut Self = self;
        let mut slice: Vec<VecZnx<&mut [u8]>> = Vec::with_capacity(len);
        for _ in 0..len {
            let (znx, new_scratch) = scratch.take_vec_znx(n, cols, size);
            scratch = new_scratch;
            slice.push(znx);
        }
        (slice, scratch)
    }

    fn take_vmp_pmat<M, B: Backend>(
        &mut self,
        module: &M,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (VmpPMat<&mut [u8], B>, &mut Self)
    where
        M: VmpPMatBytesOf + ModuleN,
    {
        let (take_slice, rem_slice) = self.take_slice(module.bytes_of_vmp_pmat(rows, cols_in, cols_out, size));
        (
            VmpPMat::from_data(take_slice, module.n(), rows, cols_in, cols_out, size),
            rem_slice,
        )
    }

    fn take_mat_znx(
        &mut self,
        n: usize, 
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (MatZnx<&mut [u8]>, &mut Self)
    {
        let (take_slice, rem_slice) = self.take_slice(MatZnx::bytes_of(n, rows, cols_in, cols_out, size));
        (
            MatZnx::from_data(take_slice, n, rows, cols_in, cols_out, size),
            rem_slice,
        )
    }
}
