//! Scratch memory allocation, borrowing, and arena-style sub-allocation.
//!
//! Provides traits for creating scratch buffers, borrowing them as
//! backend-native [`ScratchArena`] values, and carving typed layout
//! objects (e.g., [`VecZnx`], [`VecZnxDft`], [`VmpPMat`]) out of them.

use crate::{
    api::{CnvPVecBytesOf, ModuleN, SvpPPolBytesOf, VecZnxBigBytesOf, VecZnxDftBytesOf, VmpPMatBytesOf},
    layouts::{Backend, CnvPVecL, CnvPVecR, MatZnx, ScalarZnx, ScratchArena, SvpPPol, VecZnx, VecZnxBig, VecZnxDft, VmpPMat},
};

/// Allocates a new [crate::layouts::ScratchOwned] of `size` aligned bytes.
pub trait ScratchOwnedAlloc<B: Backend> {
    fn alloc(size: usize) -> Self;
}

/// Borrows an owned scratch buffer as a backend-native arena.
pub trait ScratchOwnedBorrow<B: Backend> {
    fn borrow(&mut self) -> ScratchArena<'_, B>;
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
