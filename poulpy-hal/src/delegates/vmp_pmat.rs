use crate::{
    api::{
        VmpApplyDft, VmpApplyDftTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftBackendRef, VmpApplyDftToDftTmpBytes, VmpPMatAlloc,
        VmpPMatBytesOf, VmpPrepare, VmpPrepareTmpBytes, VmpZero,
    },
    layouts::{
        Backend, MatZnxToRef, Module, ScratchArena, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef,
        VecZnxDftToMut, VecZnxDftToRef, VmpPMatBackendMut, VmpPMatBackendRef, VmpPMatOwned,
    },
    oep::HalVmpImpl,
};

macro_rules! impl_vmp_delegate {
    ($trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend + HalVmpImpl<B>,
        {
            $($body)+
        }
    };
}

impl<B: Backend> VmpPMatAlloc<B> for Module<B> {
    fn vmp_pmat_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> VmpPMatOwned<B> {
        VmpPMatOwned::alloc(self.n(), rows, cols_in, cols_out, size)
    }
}

impl<B: Backend> VmpPMatBytesOf for Module<B> {
    fn bytes_of_vmp_pmat(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        B::bytes_of_vmp_pmat(self.n(), rows, cols_in, cols_out, size)
    }
}

impl_vmp_delegate!(
    VmpPrepareTmpBytes,
    fn vmp_prepare_tmp_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        <B as HalVmpImpl<B>>::vmp_prepare_tmp_bytes(self, rows, cols_in, cols_out, size)
    }
);

impl_vmp_delegate!(
    VmpPrepare<B>,
    fn vmp_prepare<'s, A>(&self, res: &mut VmpPMatBackendMut<'_, B>, a: &A, scratch: &mut ScratchArena<'s, B>)
    where
        A: MatZnxToRef,
    {
        <B as HalVmpImpl<B>>::vmp_prepare(self, res, a, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplyDftTmpBytes,
    fn vmp_apply_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        <B as HalVmpImpl<B>>::vmp_apply_dft_tmp_bytes(self, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplyDft<B>,
    fn vmp_apply_dft<'s, R>(
        &self,
        res: &mut R,
        a: &VecZnxBackendRef<'_, B>,
        b: &VmpPMatBackendRef<'_, B>,
        scratch: &mut ScratchArena<'s, B>,
    )
    where
        R: VecZnxDftToMut<B>,
    {
        <B as HalVmpImpl<B>>::vmp_apply_dft(self, res, a, b, scratch)
    }
);

impl_vmp_delegate!(
    VmpApplyDftToDftTmpBytes,
    fn vmp_apply_dft_to_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        <B as HalVmpImpl<B>>::vmp_apply_dft_to_dft_tmp_bytes(self, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplyDftToDft<B>,
    fn vmp_apply_dft_to_dft<'s, R, A>(
        &self,
        res: &mut R,
        a: &A,
        b: &VmpPMatBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
    {
        <B as HalVmpImpl<B>>::vmp_apply_dft_to_dft(self, res, a, b, limb_offset, scratch)
    }
);

impl_vmp_delegate!(
    VmpApplyDftToDftBackendRef<B>,
    fn vmp_apply_dft_to_dft_backend_ref<'s, 'r, 'a>(
        &self,
        res: &mut VecZnxDftBackendMut<'r, B>,
        a: &VecZnxDftBackendRef<'a, B>,
        b: &VmpPMatBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        <B as HalVmpImpl<B>>::vmp_apply_dft_to_dft_backend_ref(self, res, a, b, limb_offset, scratch)
    }
);

impl_vmp_delegate!(
    VmpZero<B>,
    fn vmp_zero(&self, res: &mut VmpPMatBackendMut<'_, B>) {
        <B as HalVmpImpl<B>>::vmp_zero(self, res);
    }
);
