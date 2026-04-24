//! Scratch memory allocation, borrowing, and arena-style sub-allocation.
//!
//! Provides traits for creating scratch buffers, borrowing them as
//! backend-native [`ScratchArena`] values, and carving typed layout
//! objects (e.g., [`VecZnx`], [`VecZnxDft`], [`VmpPMat`]) out of them.

use crate::{
    api::{CnvPVecBytesOf, ModuleN, SvpPPolBytesOf, VecZnxBigBytesOf, VecZnxDftBytesOf, VmpPMatBytesOf},
    layouts::{
        Backend, CnvPVecL, CnvPVecR, MatZnx, ScalarZnx, Scratch, ScratchArena, SvpPPol, VecZnx, VecZnxBig, VecZnxDft, VmpPMat,
    },
};

/// Allocates a new [crate::layouts::ScratchOwned] of `size` aligned bytes.
pub trait ScratchOwnedAlloc<B: Backend> {
    fn alloc(size: usize) -> Self;
}

/// Borrows an owned scratch buffer as a backend-native arena.
pub trait ScratchOwnedBorrow<B: Backend> {
    fn borrow(&mut self) -> ScratchArena<'_, B>;
}

/// Wrap an array of mutable borrowed bytes into a [Scratch].
pub trait ScratchFromBytes<B: Backend> {
    fn from_bytes(data: &mut [u8]) -> &mut Scratch<B>;
}

/// Returns how many bytes left can be taken from the scratch.
pub trait ScratchAvailable {
    fn available(&self) -> usize;
}

/// Host-visible borrowed scratch region for a backend.
///
/// Device backends should not implement this unless their borrowed mutable
/// scratch region is directly accessible as a host byte slice.
pub trait HostBufMut<'a>: Sized {
    fn into_bytes(self) -> &'a mut [u8];
}

impl<'a> HostBufMut<'a> for &'a mut [u8] {
    #[inline]
    fn into_bytes(self) -> &'a mut [u8] {
        self
    }
}

/// Explicit host-only typed scalar takes from a [`ScratchArena`].
///
/// This keeps the primary arena path backend-native while still supporting
/// the CPU defaults that need raw host scratch slices.
pub trait ScratchArenaTakeHost<'a, B: Backend>: Sized {
    fn take_u8(self, len: usize) -> (&'a mut [u8], Self);

    fn take_u64(self, len: usize) -> (&'a mut [u64], Self);

    fn take_i64(self, len: usize) -> (&'a mut [i64], Self);

    fn take_i128(self, len: usize) -> (&'a mut [i128], Self);

    fn take_f64(self, len: usize) -> (&'a mut [f64], Self);
}

impl<'a, B: Backend + 'a> ScratchArenaTakeHost<'a, B> for ScratchArena<'a, B>
where
    B::BufMut<'a>: HostBufMut<'a>,
{
    #[inline]
    fn take_u8(self, len: usize) -> (&'a mut [u8], Self) {
        take_typed(self, len)
    }

    #[inline]
    fn take_u64(self, len: usize) -> (&'a mut [u64], Self) {
        take_typed(self, len)
    }

    #[inline]
    fn take_i64(self, len: usize) -> (&'a mut [i64], Self) {
        take_typed(self, len)
    }

    #[inline]
    fn take_i128(self, len: usize) -> (&'a mut [i128], Self) {
        take_typed(self, len)
    }

    #[inline]
    fn take_f64(self, len: usize) -> (&'a mut [f64], Self) {
        take_typed(self, len)
    }
}

#[inline]
fn take_typed<'a, B, T>(arena: ScratchArena<'a, B>, len: usize) -> (&'a mut [T], ScratchArena<'a, B>)
where
    B: Backend + 'a,
    B::BufMut<'a>: HostBufMut<'a>,
    T: Copy,
{
    debug_assert!(
        B::SCRATCH_ALIGN.is_multiple_of(std::mem::align_of::<T>()),
        "B::SCRATCH_ALIGN ({}) must be a multiple of align_of::<T>() ({})",
        B::SCRATCH_ALIGN,
        std::mem::align_of::<T>()
    );
    let (buf, arena) = arena.take_region(len * std::mem::size_of::<T>());
    let bytes: &'a mut [u8] = buf.into_bytes();
    let slice = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, len) };
    (slice, arena)
}

/// Takes a slice of bytes from a [Scratch] and return a new [Scratch] minus the taken array of bytes.
pub trait TakeSlice {
    fn take_slice<T>(&mut self, len: usize) -> (&mut [T], &mut Self);
}

impl<BE: Backend> Scratch<BE>
where
    Self: TakeSlice + ScratchAvailable + ScratchFromBytes<BE>,
{
    /// Splits off `len` bytes from the front and returns the taken region
    /// as a new [`Scratch`] plus the remaining scratch.
    pub fn split_at_mut(&mut self, len: usize) -> (&mut Scratch<BE>, &mut Self) {
        let (take_slice, rem_slice) = self.take_slice(len);
        (Self::from_bytes(take_slice), rem_slice)
    }

    /// Splits off `n` non-overlapping [`Scratch`] regions of `len` bytes each.
    ///
    /// # Panics
    ///
    /// Panics if `self.available() < n * len`.
    pub fn split_mut(&mut self, n: usize, len: usize) -> (Vec<&mut Scratch<BE>>, &mut Self) {
        assert!(self.available() >= n * len);
        let mut scratches: Vec<&mut Scratch<BE>> = Vec::with_capacity(n);
        let mut scratch: &mut Scratch<BE> = self;
        for _ in 0..n {
            let (tmp, scratch_new) = scratch.split_at_mut(len);
            scratch = scratch_new;
            scratches.push(tmp);
        }
        (scratches, scratch)
    }
}

impl<B: Backend> ScratchTakeBasic for Scratch<B> where Self: TakeSlice + ScratchFromBytes<B> {}

/// Backend-native arena allocation of typed HAL layouts.
///
/// This is the additive, backend-owned scratch path introduced for
/// incremental device-backend integration. It consumes a [`ScratchArena`]
/// by value and returns the carved layout together with the remaining arena.
pub trait ScratchArenaTakeBasic<'a, B: Backend>: Sized {
    /// Takes a [`CnvPVecL`] from the scratch arena.
    fn take_cnv_pvec_left<M>(self, module: &M, cols: usize, size: usize) -> (CnvPVecL<B::BufMut<'a>, B>, Self)
    where
        M: ModuleN + CnvPVecBytesOf;

    /// Takes a [`CnvPVecR`] from the scratch arena.
    fn take_cnv_pvec_right<M>(self, module: &M, cols: usize, size: usize) -> (CnvPVecR<B::BufMut<'a>, B>, Self)
    where
        M: ModuleN + CnvPVecBytesOf;

    /// Takes a [`ScalarZnx`] from the scratch arena.
    fn take_scalar_znx(self, n: usize, cols: usize) -> (ScalarZnx<B::BufMut<'a>>, Self);

    /// Takes a [`SvpPPol`] from the scratch arena.
    fn take_svp_ppol<M>(self, module: &M, cols: usize) -> (SvpPPol<B::BufMut<'a>, B>, Self)
    where
        M: SvpPPolBytesOf + ModuleN;

    /// Takes a [`VecZnx`] from the scratch arena.
    fn take_vec_znx(self, n: usize, cols: usize, size: usize) -> (VecZnx<B::BufMut<'a>>, Self);

    /// Takes a [`VecZnxBig`] from the scratch arena.
    fn take_vec_znx_big<M>(self, module: &M, cols: usize, size: usize) -> (VecZnxBig<B::BufMut<'a>, B>, Self)
    where
        M: VecZnxBigBytesOf + ModuleN;

    /// Takes a [`VecZnxDft`] from the scratch arena.
    fn take_vec_znx_dft<M>(self, module: &M, cols: usize, size: usize) -> (VecZnxDft<B::BufMut<'a>, B>, Self)
    where
        M: VecZnxDftBytesOf + ModuleN;

    /// Takes `len` consecutive [`VecZnxDft`] objects from the scratch arena.
    fn take_vec_znx_dft_slice<M>(
        self,
        module: &M,
        len: usize,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnxDft<B::BufMut<'a>, B>>, Self)
    where
        M: VecZnxDftBytesOf + ModuleN;

    /// Takes `len` consecutive [`VecZnx`] objects from the scratch arena.
    fn take_vec_znx_slice(self, len: usize, n: usize, cols: usize, size: usize) -> (Vec<VecZnx<B::BufMut<'a>>>, Self);

    /// Takes a [`VmpPMat`] from the scratch arena.
    fn take_vmp_pmat<M>(
        self,
        module: &M,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (VmpPMat<B::BufMut<'a>, B>, Self)
    where
        M: VmpPMatBytesOf + ModuleN;

    /// Takes a [`MatZnx`] from the scratch arena.
    fn take_mat_znx(self, n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> (MatZnx<B::BufMut<'a>>, Self);
}

impl<'a, B: Backend> ScratchArenaTakeBasic<'a, B> for ScratchArena<'a, B> {
    fn take_cnv_pvec_left<M>(self, module: &M, cols: usize, size: usize) -> (CnvPVecL<B::BufMut<'a>, B>, Self)
    where
        M: ModuleN + CnvPVecBytesOf,
    {
        let (data, arena) = self.take_region(module.bytes_of_cnv_pvec_left(cols, size));
        (CnvPVecL::from_data(data, module.n(), cols, size), arena)
    }

    fn take_cnv_pvec_right<M>(self, module: &M, cols: usize, size: usize) -> (CnvPVecR<B::BufMut<'a>, B>, Self)
    where
        M: ModuleN + CnvPVecBytesOf,
    {
        let (data, arena) = self.take_region(module.bytes_of_cnv_pvec_right(cols, size));
        (CnvPVecR::from_data(data, module.n(), cols, size), arena)
    }

    fn take_scalar_znx(self, n: usize, cols: usize) -> (ScalarZnx<B::BufMut<'a>>, Self) {
        let (data, arena) = self.take_region(ScalarZnx::bytes_of(n, cols));
        (ScalarZnx::from_data(data, n, cols), arena)
    }

    fn take_svp_ppol<M>(self, module: &M, cols: usize) -> (SvpPPol<B::BufMut<'a>, B>, Self)
    where
        M: SvpPPolBytesOf + ModuleN,
    {
        let (data, arena) = self.take_region(module.bytes_of_svp_ppol(cols));
        (SvpPPol::from_data(data, module.n(), cols), arena)
    }

    fn take_vec_znx(self, n: usize, cols: usize, size: usize) -> (VecZnx<B::BufMut<'a>>, Self) {
        let (data, arena) = self.take_region(VecZnx::bytes_of(n, cols, size));
        (VecZnx::from_data(data, n, cols, size), arena)
    }

    fn take_vec_znx_big<M>(self, module: &M, cols: usize, size: usize) -> (VecZnxBig<B::BufMut<'a>, B>, Self)
    where
        M: VecZnxBigBytesOf + ModuleN,
    {
        let (data, arena) = self.take_region(module.bytes_of_vec_znx_big(cols, size));
        (VecZnxBig::from_data(data, module.n(), cols, size), arena)
    }

    fn take_vec_znx_dft<M>(self, module: &M, cols: usize, size: usize) -> (VecZnxDft<B::BufMut<'a>, B>, Self)
    where
        M: VecZnxDftBytesOf + ModuleN,
    {
        let (data, arena) = self.take_region(module.bytes_of_vec_znx_dft(cols, size));
        (VecZnxDft::from_data(data, module.n(), cols, size), arena)
    }

    fn take_vec_znx_dft_slice<M>(
        self,
        module: &M,
        len: usize,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnxDft<B::BufMut<'a>, B>>, Self)
    where
        M: VecZnxDftBytesOf + ModuleN,
    {
        let mut arena: Self = self;
        let mut slice: Vec<VecZnxDft<B::BufMut<'a>, B>> = Vec::with_capacity(len);
        for _ in 0..len {
            let (znx, rem) = arena.take_vec_znx_dft(module, cols, size);
            arena = rem;
            slice.push(znx);
        }
        (slice, arena)
    }

    fn take_vec_znx_slice(self, len: usize, n: usize, cols: usize, size: usize) -> (Vec<VecZnx<B::BufMut<'a>>>, Self) {
        let mut arena: Self = self;
        let mut slice: Vec<VecZnx<B::BufMut<'a>>> = Vec::with_capacity(len);
        for _ in 0..len {
            let (znx, rem) = arena.take_vec_znx(n, cols, size);
            arena = rem;
            slice.push(znx);
        }
        (slice, arena)
    }

    fn take_vmp_pmat<M>(
        self,
        module: &M,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (VmpPMat<B::BufMut<'a>, B>, Self)
    where
        M: VmpPMatBytesOf + ModuleN,
    {
        let (data, arena) = self.take_region(module.bytes_of_vmp_pmat(rows, cols_in, cols_out, size));
        (VmpPMat::from_data(data, module.n(), rows, cols_in, cols_out, size), arena)
    }

    fn take_mat_znx(self, n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> (MatZnx<B::BufMut<'a>>, Self) {
        let (data, arena) = self.take_region(MatZnx::bytes_of(n, rows, cols_in, cols_out, size));
        (MatZnx::from_data(data, n, rows, cols_in, cols_out, size), arena)
    }
}

/// Arena-style sub-allocation of typed layout objects from a [`Scratch`] buffer.
///
/// Each `take_*` method carves the required number of bytes from the front
/// of the scratch, constructs the layout object over that memory, and returns
/// the object together with the remaining scratch.
pub trait ScratchTakeBasic
where
    Self: TakeSlice,
{
    /// Takes a [`CnvPVecL`] from the scratch.
    fn take_cnv_pvec_left<M, B: Backend>(&mut self, module: &M, cols: usize, size: usize) -> (CnvPVecL<&mut [u8], B>, &mut Self)
    where
        M: ModuleN + CnvPVecBytesOf,
    {
        let (take_slice, rem_slice) = self.take_slice(module.bytes_of_cnv_pvec_left(cols, size));
        (CnvPVecL::from_data(take_slice, module.n(), cols, size), rem_slice)
    }

    /// Takes a [`CnvPVecR`] from the scratch.
    fn take_cnv_pvec_right<M, B: Backend>(&mut self, module: &M, cols: usize, size: usize) -> (CnvPVecR<&mut [u8], B>, &mut Self)
    where
        M: ModuleN + CnvPVecBytesOf,
    {
        let (take_slice, rem_slice) = self.take_slice(module.bytes_of_cnv_pvec_right(cols, size));
        (CnvPVecR::from_data(take_slice, module.n(), cols, size), rem_slice)
    }

    /// Takes a [`ScalarZnx`] from the scratch.
    fn take_scalar_znx(&mut self, n: usize, cols: usize) -> (ScalarZnx<&mut [u8]>, &mut Self) {
        let (take_slice, rem_slice) = self.take_slice(ScalarZnx::bytes_of(n, cols));
        (ScalarZnx::from_data(take_slice, n, cols), rem_slice)
    }

    /// Takes a [`SvpPPol`] from the scratch.
    fn take_svp_ppol<M, B: Backend>(&mut self, module: &M, cols: usize) -> (SvpPPol<&mut [u8], B>, &mut Self)
    where
        M: SvpPPolBytesOf + ModuleN,
    {
        let (take_slice, rem_slice) = self.take_slice(module.bytes_of_svp_ppol(cols));
        (SvpPPol::from_data(take_slice, module.n(), cols), rem_slice)
    }

    /// Takes a [`VecZnx`] from the scratch.
    fn take_vec_znx(&mut self, n: usize, cols: usize, size: usize) -> (VecZnx<&mut [u8]>, &mut Self) {
        let (take_slice, rem_slice) = self.take_slice(VecZnx::bytes_of(n, cols, size));
        (VecZnx::from_data(take_slice, n, cols, size), rem_slice)
    }

    /// Takes a [`VecZnxBig`] from the scratch.
    fn take_vec_znx_big<M, B: Backend>(&mut self, module: &M, cols: usize, size: usize) -> (VecZnxBig<&mut [u8], B>, &mut Self)
    where
        M: VecZnxBigBytesOf + ModuleN,
    {
        let (take_slice, rem_slice) = self.take_slice(module.bytes_of_vec_znx_big(cols, size));
        (VecZnxBig::from_data(take_slice, module.n(), cols, size), rem_slice)
    }

    /// Takes a [`VecZnxDft`] from the scratch.
    fn take_vec_znx_dft<M, B: Backend>(&mut self, module: &M, cols: usize, size: usize) -> (VecZnxDft<&mut [u8], B>, &mut Self)
    where
        M: VecZnxDftBytesOf + ModuleN,
    {
        let (take_slice, rem_slice) = self.take_slice(module.bytes_of_vec_znx_dft(cols, size));

        (VecZnxDft::from_data(take_slice, module.n(), cols, size), rem_slice)
    }

    /// Takes `len` consecutive [`VecZnxDft`] objects from the scratch.
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

    /// Takes `len` consecutive [`VecZnx`] objects from the scratch.
    fn take_vec_znx_slice(&mut self, len: usize, n: usize, cols: usize, size: usize) -> (Vec<VecZnx<&mut [u8]>>, &mut Self) {
        let mut scratch: &mut Self = self;
        let mut slice: Vec<VecZnx<&mut [u8]>> = Vec::with_capacity(len);
        for _ in 0..len {
            let (znx, new_scratch) = scratch.take_vec_znx(n, cols, size);
            scratch = new_scratch;
            slice.push(znx);
        }
        (slice, scratch)
    }

    /// Takes a [`VmpPMat`] from the scratch.
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

    /// Takes a [`MatZnx`] from the scratch.
    fn take_mat_znx(
        &mut self,
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (MatZnx<&mut [u8]>, &mut Self) {
        let (take_slice, rem_slice) = self.take_slice(MatZnx::bytes_of(n, rows, cols_in, cols_out, size));
        (MatZnx::from_data(take_slice, n, rows, cols_in, cols_out, size), rem_slice)
    }
}
