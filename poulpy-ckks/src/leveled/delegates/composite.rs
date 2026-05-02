use anyhow::{Result, bail, ensure};
use poulpy_core::{
    GLWETensoring, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWETensor, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        ModuleCoreAlloc, TorusPrecision,
    },
};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Data, Module, ScratchArena},
};

use crate::{
    CKKSInfos, CKKSPlaintexToBackendRef,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, ciphertext::CKKSOffset},
    leveled::api::{
        CKKSAddManyOps, CKKSAddOps, CKKSAddOpsUnsafe, CKKSDotProductOps, CKKSMulAddOps, CKKSMulManyOps, CKKSMulOps,
        CKKSMulSubOps, CKKSRescaleOps, CKKSSubOps,
    },
    oep::CKKSImpl,
};

fn take_mul_tmp<BE: Backend, D: Data>(module: &Module<BE>, dst: &CKKSCiphertext<D>) -> CKKSCiphertext<BE::OwnedBuf>
where
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
{
    module.ckks_ciphertext_alloc_from_infos(dst)
}

fn ensure_accumulation_fits<D: Data>(op: &'static str, dst: &CKKSCiphertext<D>, n: usize) -> Result<()> {
    let base2k: usize = dst.base2k().as_usize();
    ensure!(base2k < 64, "{op}: unsupported base2k={base2k}");
    ensure!(
        n <= (1usize << (63 - base2k)),
        "{op}: {n} terms risks i64 overflow at base2k={base2k}",
    );
    Ok(())
}

// --- CKKSAddManyOps ---

impl<BE: Backend + CKKSImpl<BE>> CKKSAddManyOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE> + CKKSRescaleOps<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_add_many_tmp_bytes(&self) -> usize {
        self.ckks_add_tmp_bytes()
    }

    fn ckks_add_many<Dst: Data, Src: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        inputs: &[&CKKSCiphertext<Src>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: GLWEToBackendRef<BE>,
    {
        match inputs.len() {
            0 => bail!("ckks_add_many: inputs must contain at least one ciphertext"),
            1 => {
                self.ckks_rescale_into(dst, dst.offset_unary(inputs[0]), inputs[0], scratch)?;
            }
            _ => {
                ensure_accumulation_fits("ckks_add_many", dst, inputs.len())?;
                self.ckks_add_into(dst, inputs[0], inputs[1], scratch)?;
                for ct in &inputs[2..] {
                    self.ckks_add_assign(dst, *ct, scratch)?;
                }
            }
        }
        Ok(())
    }
}

// --- CKKSMulManyOps ---

fn ceil_log2(n: usize) -> usize {
    debug_assert!(n >= 1);
    if n <= 1 { 0 } else { (n - 1).ilog2() as usize + 1 }
}

impl<BE: Backend + CKKSImpl<BE>> CKKSMulManyOps<BE> for Module<BE>
where
    Module<BE>: CKKSMulOps<BE> + CKKSRescaleOps<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_mul_many_tmp_bytes<R, T>(&self, n: usize, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        T: GGLWEInfos,
    {
        let mul_scratch: usize = self.ckks_mul_tmp_bytes(res, tsk);
        if n <= 2 {
            return mul_scratch;
        }
        let depth: usize = ceil_log2(n);
        2 * depth * GLWE::<Vec<u8>>::bytes_of_from_infos(res) + mul_scratch
    }

    fn ckks_mul_many<Dst: Data, Src: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        inputs: &[&CKKSCiphertext<Src>],
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<Src>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
    {
        if inputs.is_empty() {
            bail!("ckks_mul_many: inputs must contain at least one ciphertext");
        }
        anyhow::ensure!(
            inputs.iter().all(|c| c.log_delta() == inputs[0].log_delta()),
            "ckks_mul_many: all inputs must have the same log_delta"
        );
        if inputs.len() == 1 {
            self.ckks_rescale_into(dst, dst.offset_unary(inputs[0]), inputs[0], scratch)?;
            return Ok(());
        }

        let mut acc = take_mul_tmp(self, inputs[0]);
        self.ckks_mul_into(&mut acc, inputs[0], inputs[1], tsk, scratch)?;

        for ct in &inputs[2..] {
            let mut compact = self.ckks_ciphertext_alloc(acc.base2k(), acc.effective_k().into());
            self.ckks_rescale_into(&mut compact, 0, &acc, scratch)?;

            let mut next = take_mul_tmp(self, inputs[0]);
            self.ckks_mul_into(&mut next, &compact, *ct, tsk, scratch)?;
            acc = next;
        }

        self.ckks_rescale_into(dst, dst.offset_unary(&acc), &acc, scratch)
    }
}

// --- CKKSMulAddOps ---

impl<BE: Backend + CKKSImpl<BE>> CKKSMulAddOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE> + CKKSMulOps<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_mul_add_ct_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        T: GGLWEInfos,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_tmp_bytes(res, tsk).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_mul_add_pt_vec_znx_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        A: GLWEInfos + CKKSInfos,
        P: CKKSInfos,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_znx_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_mul_add_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        A: GLWEInfos + CKKSInfos,
        P: CKKSInfos,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_const_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_mul_add_ct_into<Dst: Data, A: Data, B: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<B>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
    {
        let mut tmp = take_mul_tmp(self, dst);
        self.ckks_mul_into(&mut tmp, a, b, tsk, scratch)?;
        self.ckks_add_assign(dst, &tmp, scratch)
    }

    fn ckks_mul_add_pt_vec_znx_into<Dst: Data, A: Data, P>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
    {
        let mut tmp = take_mul_tmp(self, dst);
        self.ckks_mul_pt_vec_znx_into(&mut tmp, a, pt_znx, scratch)?;
        self.ckks_add_assign(dst, &tmp, scratch)
    }

    fn ckks_mul_add_pt_const_znx_into<Dst: Data, A: Data, P>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
    {
        let mut tmp = take_mul_tmp(self, dst);
        self.ckks_mul_pt_const_znx_into(&mut tmp, a, pt_znx, scratch)?;
        self.ckks_add_assign(dst, &tmp, scratch)
    }
}

// --- CKKSMulSubOps ---

impl<BE: Backend + CKKSImpl<BE>> CKKSMulSubOps<BE> for Module<BE>
where
    Module<BE>: CKKSSubOps<BE> + CKKSMulOps<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_mul_sub_ct_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        T: GGLWEInfos,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_tmp_bytes(res, tsk).max(self.ckks_sub_tmp_bytes())
    }

    fn ckks_mul_sub_pt_vec_znx_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        A: GLWEInfos + CKKSInfos,
        P: CKKSInfos,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_znx_tmp_bytes(res, a, b).max(self.ckks_sub_tmp_bytes())
    }

    fn ckks_mul_sub_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        A: GLWEInfos + CKKSInfos,
        P: CKKSInfos,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_const_tmp_bytes(res, a, b).max(self.ckks_sub_tmp_bytes())
    }

    fn ckks_mul_sub_ct_into<Dst: Data, A: Data, B: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<B>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
    {
        let mut tmp = take_mul_tmp(self, dst);
        self.ckks_mul_into(&mut tmp, a, b, tsk, scratch)?;
        self.ckks_sub_assign(dst, &tmp, scratch)
    }

    fn ckks_mul_sub_pt_vec_znx_into<Dst: Data, A: Data, P>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
    {
        let mut tmp = take_mul_tmp(self, dst);
        self.ckks_mul_pt_vec_znx_into(&mut tmp, a, pt_znx, scratch)?;
        self.ckks_sub_assign(dst, &tmp, scratch)
    }

    fn ckks_mul_sub_pt_const_znx_into<Dst: Data, A: Data, P>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
    {
        let mut tmp = take_mul_tmp(self, dst);
        self.ckks_mul_pt_const_znx_into(&mut tmp, a, pt_znx, scratch)?;
        self.ckks_sub_assign(dst, &tmp, scratch)
    }
}

// --- CKKSDotProductOps ---

fn check_lengths(op: &'static str, a_len: usize, b_len: usize) -> Result<()> {
    if a_len == 0 {
        bail!("{op}: inputs must contain at least one pair");
    }
    if a_len != b_len {
        bail!("{op}: length mismatch between ct vector ({a_len}) and weight vector ({b_len})");
    }
    Ok(())
}

fn accumulate_unnormalized<BE, D, F>(
    module: &Module<BE>,
    dst: &mut CKKSCiphertext<D>,
    n: usize,
    scratch: &mut ScratchArena<'_, BE>,
    mut mul_term_into_tmp: F,
) -> Result<()>
where
    BE: Backend + CKKSImpl<BE>,
    D: Data,
    Module<BE>: CKKSAddOps<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
    CKKSCiphertext<D>: GLWEToBackendMut<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    F: FnMut(&mut CKKSCiphertext<BE::OwnedBuf>, usize, &mut ScratchArena<'_, BE>) -> Result<()>,
{
    if n <= 1 {
        return Ok(());
    }
    let mut tmp = take_mul_tmp(module, dst);
    for i in 1..n {
        mul_term_into_tmp(&mut tmp, i, scratch)?;
        <Module<BE> as CKKSAddOps<BE>>::ckks_add_assign(module, dst, &tmp, scratch)?;
    }
    Ok(())
}

impl<BE: Backend + CKKSImpl<BE>> CKKSDotProductOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE>
        + CKKSAddOpsUnsafe<BE>
        + CKKSMulOps<BE>
        + CKKSRescaleOps<BE>
        + GLWETensoring<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_dot_product_ct_tmp_bytes<R, T>(&self, n: usize, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        T: GGLWEInfos,
    {
        let mul_scratch: usize = self.ckks_mul_tmp_bytes(res, tsk);
        if n <= 1 {
            return mul_scratch;
        }
        let ct_bytes: usize = GLWE::<Vec<u8>>::bytes_of_from_infos(res);
        let fallback: usize = ct_bytes + mul_scratch.max(self.ckks_add_tmp_bytes());
        let tensor_layout = GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: TorusPrecision(res.max_k().as_u32()),
            rank: res.rank(),
        };
        let tensor_bytes: usize = GLWETensor::bytes_of_from_infos(&tensor_layout);
        let inner: usize = self
            .ckks_rescale_tmp_bytes()
            .max(self.glwe_tensor_apply_tmp_bytes(&tensor_layout, res, res))
            .max(self.glwe_tensor_relinearize_tmp_bytes(res, &tensor_layout, tsk));
        let fast: usize = 2 * n * ct_bytes + tensor_bytes + inner;
        fallback.max(fast)
    }

    fn ckks_dot_product_pt_vec_znx_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        A: GLWEInfos + CKKSInfos,
        P: CKKSInfos,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_znx_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_dot_product_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        A: GLWEInfos + CKKSInfos,
        P: CKKSInfos,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_const_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_dot_product_ct<Dst: Data, D: Data, E: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSCiphertext<E>],
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<E>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
    {
        check_lengths("ckks_dot_product_ct", a.len(), b.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_ct", dst, n)?;
        self.ckks_mul_into(dst, a[0], b[0], tsk, scratch)?;
        accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| self.ckks_mul_into(tmp, a[i], b[i], tsk, s))
    }

    fn ckks_dot_product_pt_vec_znx<Dst: Data, D: Data, E>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&E],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + GLWEInfos,
        E: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
    {
        check_lengths("ckks_dot_product_pt_vec_znx", a.len(), b.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_pt_vec_znx", dst, n)?;
        self.ckks_mul_pt_vec_znx_into(dst, a[0], b[0], scratch)?;
        accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| {
            self.ckks_mul_pt_vec_znx_into(tmp, a[i], b[i], s)
        })
    }

    fn ckks_dot_product_pt_const_znx<Dst: Data, D: Data, E>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&E],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + GLWEInfos,
        E: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
    {
        check_lengths("ckks_dot_product_pt_const_znx", a.len(), b.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_pt_const_znx", dst, n)?;
        self.ckks_mul_pt_const_znx_into(dst, a[0], b[0], scratch)?;
        accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| {
            self.ckks_mul_pt_const_znx_into(tmp, a[i], b[i], s)
        })
    }
}
