use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftZero, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes, VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDftBackendRef, VmpApplyDftToDftTmpBytes,
    },
    layouts::{
        Backend, Module, ScratchArena, VecZnxBig, VecZnxBigReborrowBackendRef, VecZnxDft, VecZnxDftReborrowBackendRef, ZnxInfos,
    },
};

pub use crate::api::GLWEExternalProduct;
use crate::api::GLWEExternalProductInternal;
use crate::{
    GLWENormalize, ScratchArenaTakeCore,
    layouts::{
        GGSWInfos, GGSWPreparedBackendRef, GLWEBackendMut, GLWEBackendRef, GLWEInfos, GLWELayout, LWEInfos,
        glwe_backend_ref_from_mut, prepared::GGSWPreparedToBackendRef,
    },
};

fn glwe_external_product_dft_fill<'s, 'r, 'a, BE, M>(
    module: &M,
    res_dft: &mut VecZnxDft<BE::BufMut<'r>, BE>,
    a: &GLWEBackendRef<'a, BE>,
    ggsw: &GGSWPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxNormalizeTmpBytes
        + VecZnxDftApply<BE>
        + VmpApplyDftToDftBackendRef<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxDftZero<BE>,
    for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
{
    let cols: usize = (ggsw.rank() + 1).into();
    let dsize: usize = ggsw.dsize().into();
    let a_size: usize = a.size();
    scratch.scope(|scratch_phase| {
        let (mut a_dft, mut scratch_1) = scratch_phase.take_vec_znx_dft(module, cols, a_size.div_ceil(dsize));
        for col in 0..a_dft.cols() {
            module.vec_znx_dft_zero(&mut a_dft, col);
        }

        if dsize == 1 {
            a_dft.size = a_size;
            res_dft.size = ggsw.size();
            for j in 0..cols {
                module.vec_znx_dft_apply(1, 0, &mut a_dft, j, &a.data, j);
            }
            let a_dft_ref = a_dft.reborrow_backend_ref();
            module.vmp_apply_dft_to_dft_backend_ref(res_dft, &a_dft_ref, &ggsw.data, 0, &mut scratch_1.borrow());
        } else {
            let (mut res_dft_tmp, mut scratch_2) = scratch_1.take_vec_znx_dft(module, res_dft.cols(), ggsw.size());

            for di in 0..dsize {
                a_dft.size = (a.size() + di) / dsize;
                res_dft.size = ggsw.size() - ((dsize - di) as isize - 2).max(0) as usize;

                for j in 0..cols {
                    module.vec_znx_dft_apply(dsize, dsize - 1 - di, &mut a_dft, j, &a.data, j);
                }

                if di == 0 {
                    let a_dft_ref = a_dft.reborrow_backend_ref();
                    module.vmp_apply_dft_to_dft_backend_ref(res_dft, &a_dft_ref, &ggsw.data, 0, &mut scratch_2.borrow());
                } else {
                    res_dft_tmp.size = res_dft.size();
                    let a_dft_ref = a_dft.reborrow_backend_ref();
                    module.vmp_apply_dft_to_dft_backend_ref(
                        &mut res_dft_tmp,
                        &a_dft_ref,
                        &ggsw.data,
                        di,
                        &mut scratch_2.borrow(),
                    );
                    let res_dft_tmp_ref = res_dft_tmp.reborrow_backend_ref();
                    for col in 0..cols {
                        module.vec_znx_dft_add_assign(res_dft, col, &res_dft_tmp_ref, col);
                    }
                }
            }
        }
    });
}

fn glwe_external_product_internal_fill<'s, 'r, 'b, 'a, BE, M>(
    module: &M,
    res_big: &mut VecZnxBig<BE::BufMut<'b>, BE>,
    res_dft: &mut VecZnxDft<BE::BufMut<'r>, BE>,
    a: &GLWEBackendRef<'a, BE>,
    ggsw: &GGSWPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxNormalizeTmpBytes
        + VecZnxDftApply<BE>
        + VmpApplyDftToDftBackendRef<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxDftZero<BE>,
    for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
{
    glwe_external_product_dft_fill(module, res_dft, a, ggsw, scratch);
    let cols: usize = (ggsw.rank() + 1).into();
    let res_dft_ref = res_dft.reborrow_backend_ref();
    for col in 0..cols {
        module.vec_znx_idft_apply(res_big, col, &res_dft_ref, col, &mut scratch.borrow());
    }
}

pub(crate) trait GLWEExternalProductDefault<BE: Backend>:
    Sized
    + GLWEExternalProductInternal<BE>
    + VecZnxDftBytesOf
    + VecZnxDftZero<BE>
    + VmpApplyDftToDftTmpBytes
    + VecZnxDftApply<BE>
    + VmpApplyDftToDftBackendRef<BE>
    + VecZnxDftAddAssign<BE>
    + VecZnxBigBytesOf
    + VecZnxIdftApply<BE>
    + VecZnxBigNormalize<BE>
    + VecZnxBigNormalizeTmpBytes
    + VecZnxBigAddSmallAssign<BE>
    + VecZnxIdftApplyTmpBytes
    + GLWENormalize<BE>
where
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn glwe_external_product_dft_fill_tmp_bytes<A, B>(&self, a_infos: &A, ggsw_infos: &B) -> usize
    where
        A: GLWEInfos,
        B: GGSWInfos,
    {
        let align: usize = BE::SCRATCH_ALIGN;
        let in_size: usize = a_infos
            .max_k()
            .div_ceil(ggsw_infos.base2k())
            .div_ceil(ggsw_infos.dsize().into()) as usize;
        let ggsw_size: usize = ggsw_infos.size();
        let cols: usize = (ggsw_infos.rank() + 1).into();
        let lvl_0: usize = self.bytes_of_vec_znx_dft(cols, in_size);
        let lvl_1: usize = if ggsw_infos.dsize() > 1 {
            self.bytes_of_vec_znx_dft(cols, ggsw_size)
        } else {
            0
        };
        let lvl_2: usize = self.vmp_apply_dft_to_dft_tmp_bytes(ggsw_size, in_size, in_size, cols, cols, ggsw_size);

        lvl_0.next_multiple_of(align) + lvl_1.next_multiple_of(align) + lvl_2
    }

    fn glwe_external_product_tmp_bytes_default<R, A, B>(&self, res: &R, a: &A, ggsw: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos,
    {
        let align: usize = BE::SCRATCH_ALIGN;
        let cols: usize = res.rank().as_usize() + 1;
        let lvl_0: usize = self.bytes_of_vec_znx_dft(cols, ggsw.size());
        let lvl_1: usize = self.bytes_of_vec_znx_big(cols, ggsw.size()).next_multiple_of(align)
            + self
                .vec_znx_idft_apply_tmp_bytes()
                .max(self.vec_znx_big_normalize_tmp_bytes());
        let lvl_2: usize = if a.base2k() != ggsw.base2k() {
            let a_conv_infos = GLWELayout {
                n: a.n(),
                base2k: ggsw.base2k(),
                k: a.max_k(),
                rank: a.rank(),
            };
            let lvl_2_0: usize = crate::layouts::GLWE::<Vec<u8>>::bytes_of_from_infos(&a_conv_infos);
            let lvl_2_1: usize = self
                .glwe_normalize_tmp_bytes()
                .max(self.glwe_external_product_dft_fill_tmp_bytes(&a_conv_infos, ggsw));
            lvl_2_0 + lvl_2_1
        } else {
            self.glwe_external_product_internal_tmp_bytes(res, a, ggsw)
        };

        lvl_0.next_multiple_of(align) + lvl_1.max(lvl_2)
    }

    fn glwe_external_product_inplace_default<'s, 'r, D>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        ggsw: &D,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        D: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's,
    {
        assert_eq!(ggsw.rank(), res.rank());
        assert_eq!(ggsw.n(), res.n());
        assert!(
            scratch.available() >= self.glwe_external_product_tmp_bytes_default(res, res, ggsw),
            "scratch.available(): {} < GLWEExternalProduct::glwe_external_product_tmp_bytes: {}",
            scratch.available(),
            self.glwe_external_product_tmp_bytes_default(res, res, ggsw)
        );

        let res_base2k: usize = res.base2k().as_usize();
        let ggsw_base2k: usize = ggsw.base2k().as_usize();
        let ggsw: GGSWPreparedBackendRef<'_, BE> = ggsw.to_backend_ref();
        let cols: usize = (res.rank() + 1).into();
        let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft(self, (res.rank() + 1).into(), ggsw.size());
        for col in 0..res_dft.cols() {
            self.vec_znx_dft_zero(&mut res_dft, col);
        }

        let mut scratch = scratch_1;
        if res_base2k != ggsw_base2k {
            scratch.scope(|scratch_phase| {
                let (mut res_conv, mut scratch_2) = scratch_phase.take_glwe(&GLWELayout {
                    n: res.n(),
                    base2k: ggsw.base2k(),
                    k: res.max_k(),
                    rank: res.rank(),
                });
                self.glwe_normalize(
                    &mut res_conv,
                    &glwe_backend_ref_from_mut::<BE>(&*res),
                    &mut scratch_2.borrow(),
                );
                glwe_external_product_dft_fill(
                    self,
                    &mut res_dft,
                    &glwe_backend_ref_from_mut::<BE>(&res_conv),
                    &ggsw,
                    &mut scratch_2,
                );
            });
        } else {
            glwe_external_product_dft_fill(
                self,
                &mut res_dft,
                &glwe_backend_ref_from_mut::<BE>(&*res),
                &ggsw,
                &mut scratch.borrow(),
            );
        }

        let (mut res_big, mut scratch) = scratch.borrow().take_vec_znx_big(self, cols, res_dft.size);
        let res_dft_ref = res_dft.reborrow_backend_ref();
        for col in 0..cols {
            self.vec_znx_idft_apply(&mut res_big, col, &res_dft_ref, col, &mut scratch.borrow());
        }
        let res_big_ref = res_big.reborrow_backend_ref();
        for j in 0..cols {
            self.vec_znx_big_normalize(
                &mut res.data,
                res_base2k,
                0,
                j,
                &res_big_ref,
                ggsw_base2k,
                j,
                &mut scratch.borrow(),
            );
        }
    }

    fn glwe_external_product_default<'s, 'r, 'a, G>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &GLWEBackendRef<'a, BE>,
        ggsw: &G,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        G: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's,
    {
        assert_eq!(ggsw.rank(), a.rank());
        assert_eq!(ggsw.rank(), res.rank());
        assert_eq!(ggsw.n(), res.n());
        assert_eq!(a.n(), res.n());
        assert!(
            scratch.available() >= self.glwe_external_product_tmp_bytes_default(res, a, ggsw),
            "scratch.available(): {} < GLWEExternalProduct::glwe_external_product_tmp_bytes: {}",
            scratch.available(),
            self.glwe_external_product_tmp_bytes_default(res, a, ggsw)
        );

        let a_base2k: usize = a.base2k().into();
        let ggsw_base2k: usize = ggsw.base2k().into();
        let res_base2k: usize = res.base2k().into();
        let ggsw: GGSWPreparedBackendRef<'_, BE> = ggsw.to_backend_ref();
        let cols: usize = (res.rank() + 1).into();
        let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft(self, (res.rank() + 1).into(), ggsw.size());
        for col in 0..res_dft.cols() {
            self.vec_znx_dft_zero(&mut res_dft, col);
        }

        let mut scratch = scratch_1;
        if a_base2k != ggsw_base2k {
            scratch.scope(|scratch_phase| {
                let (mut a_conv, mut scratch_2) = scratch_phase.take_glwe(&GLWELayout {
                    n: a.n(),
                    base2k: ggsw.base2k(),
                    k: a.max_k(),
                    rank: a.rank(),
                });
                self.glwe_normalize(&mut a_conv, a, &mut scratch_2.borrow());
                glwe_external_product_dft_fill(
                    self,
                    &mut res_dft,
                    &glwe_backend_ref_from_mut::<BE>(&a_conv),
                    &ggsw,
                    &mut scratch_2,
                );
            });
        } else {
            glwe_external_product_dft_fill(self, &mut res_dft, a, &ggsw, &mut scratch.borrow());
        }

        let (mut res_big, mut scratch) = scratch.borrow().take_vec_znx_big(self, cols, res_dft.size);
        let res_dft_ref = res_dft.reborrow_backend_ref();
        for col in 0..cols {
            self.vec_znx_idft_apply(&mut res_big, col, &res_dft_ref, col, &mut scratch.borrow());
        }
        let res_big_ref = res_big.reborrow_backend_ref();
        for j in 0..cols {
            self.vec_znx_big_normalize(
                &mut res.data,
                res_base2k,
                0,
                j,
                &res_big_ref,
                ggsw_base2k,
                j,
                &mut scratch.borrow(),
            );
        }
    }
}

impl<BE: Backend> GLWEExternalProductDefault<BE> for Module<BE>
where
    Self: GLWEExternalProductInternal<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VmpApplyDftToDftTmpBytes
        + VecZnxDftApply<BE>
        + VmpApplyDftToDftBackendRef<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxIdftApplyTmpBytes
        + GLWENormalize<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
}

impl<BE: Backend> GLWEExternalProductInternal<BE> for Module<BE>
where
    Self: ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxNormalizeTmpBytes
        + VecZnxDftApply<BE>
        + VmpApplyDftToDftBackendRef<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE>
        + VecZnxNormalize<BE>
        + VecZnxDftZero<BE>,
{
    fn glwe_external_product_internal_tmp_bytes<R, A, B>(&self, _res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos,
    {
        let align: usize = BE::SCRATCH_ALIGN;
        let in_size: usize = a_infos.max_k().div_ceil(b_infos.base2k()).div_ceil(b_infos.dsize().into()) as usize;
        let ggsw_size: usize = b_infos.size();
        let cols: usize = (b_infos.rank() + 1).into();
        let lvl_0: usize = self.bytes_of_vec_znx_dft(cols, in_size);
        let lvl_1: usize = if b_infos.dsize() > 1 {
            self.bytes_of_vec_znx_dft(cols, ggsw_size)
        } else {
            0
        };
        let lvl_2: usize = self.vmp_apply_dft_to_dft_tmp_bytes(ggsw_size, in_size, in_size, cols, cols, ggsw_size);
        let lvl_3: usize =
            self.bytes_of_vec_znx_big(cols, ggsw_size).next_multiple_of(align) + self.vec_znx_idft_apply_tmp_bytes();
        (lvl_0.next_multiple_of(align) + lvl_1.next_multiple_of(align) + lvl_2).max(lvl_3)
    }

    fn glwe_external_product_dft<'s, 'r, 'a, G>(
        &self,
        res_dft: &mut VecZnxDft<BE::BufMut<'r>, BE>,
        a: &GLWEBackendRef<'a, BE>,
        ggsw: &G,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        G: GGSWPreparedToBackendRef<BE>,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
        BE: 's,
    {
        let ggsw: GGSWPreparedBackendRef<'_, BE> = ggsw.to_backend_ref();
        glwe_external_product_dft_fill(self, res_dft, a, &ggsw, scratch);
    }

    fn glwe_external_product_internal<'s, 'r, 'a, G>(
        &self,
        mut res_dft: VecZnxDft<BE::BufMut<'r>, BE>,
        a: &GLWEBackendRef<'a, BE>,
        ggsw: &G,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        G: GGSWPreparedToBackendRef<BE>,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
        BE: 's,
    {
        let ggsw: GGSWPreparedBackendRef<'_, BE> = ggsw.to_backend_ref();

        assert_eq!(a.base2k(), ggsw.base2k());
        assert!(
            scratch.available() >= self.glwe_external_product_internal_tmp_bytes(&ggsw, a, &ggsw),
            "scratch.available(): {} < GLWEExternalProductInternal::glwe_external_product_internal_tmp_bytes: {}",
            scratch.available(),
            self.glwe_external_product_internal_tmp_bytes(&ggsw, a, &ggsw)
        );

        let cols: usize = (ggsw.rank() + 1).into();
        let (mut res_big, mut scratch_1) = scratch.borrow().take_vec_znx_big(self, cols, ggsw.size());
        glwe_external_product_internal_fill(self, &mut res_big, &mut res_dft, a, &ggsw, &mut scratch_1.borrow());
    }
}
