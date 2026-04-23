//! CKKS inner product `dst = Σ aᵢ · bᵢ`.

use anyhow::{Result, bail, ensure};
use poulpy_core::{
    GLWEAdd, GLWEMulConst, GLWEMulPlain, GLWENormalize, GLWERotate, GLWEShift, GLWETensoring, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWETensor, GLWETensorKeyPrepared, GLWEToMut, GLWEToRef, LWEInfos,
        TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxAddAssign, VecZnxRshAddInto},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{
    CKKSInfos, CKKSMeta, checked_log_hom_rem_sub, checked_mul_ct_log_hom_rem,
    layouts::{
        CKKSCiphertext,
        plaintext::{
            CKKSConstPlaintextConversion, CKKSPlaintextConversion, CKKSPlaintextCstRnx, CKKSPlaintextCstZnx, CKKSPlaintextVecRnx,
            CKKSPlaintextVecZnx,
        },
    },
    leveled::{
        operations::{
            add::{CKKSAddOps, CKKSAddOpsWithoutNormalization},
            mul::CKKSMulOps,
        },
        rescale::CKKSRescaleOps,
    },
    oep::CKKSImpl,
};

pub trait CKKSDotProductOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_dot_product_ct_tmp_bytes<R, T>(&self, n: usize, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWEShift<BE> + GLWETensoring<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>;

    fn ckks_dot_product_pt_vec_znx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>;

    fn ckks_dot_product_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>;

    fn ckks_dot_product_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>;

    fn ckks_dot_product_ct<D: DataRef, E: DataRef>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSCiphertext<E>],
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + GLWETensoring<BE>
            + VecZnxAddAssign
            + CKKSAddOpsWithoutNormalization<BE>
            + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_dot_product_pt_vec_znx<D: DataRef, E: DataRef>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextVecZnx<E>],
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd
            + GLWEMulPlain<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + VecZnxRshAddInto<BE>
            + CKKSAddOpsWithoutNormalization<BE>
            + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_dot_product_pt_vec_rnx<D: DataRef, F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextVecRnx<F>],
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN
            + GLWEAdd
            + GLWEMulPlain<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + VecZnxRshAddInto<BE>
            + CKKSAddOpsWithoutNormalization<BE>
            + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_dot_product_const_znx<D: DataRef>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextCstZnx],
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + CKKSAddOpsWithoutNormalization<BE>
            + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_dot_product_const_rnx<D: DataRef, F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextCstRnx<F>],
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + CKKSAddOpsWithoutNormalization<BE>
            + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;
}

fn check_lengths(op: &'static str, a_len: usize, b_len: usize) -> Result<()> {
    if a_len == 0 {
        bail!("{op}: inputs must contain at least one pair");
    }
    if a_len != b_len {
        bail!("{op}: length mismatch between ct vector ({a_len}) and weight vector ({b_len})");
    }
    Ok(())
}

/// Overflow guard for the unnormalized accumulation. Starting from
/// K-normalized summands each contributes ≤ 2^(base2k-1) per limb; i64
/// overflow requires `n · 2^(base2k-1) ≤ 2^63`. See §3.3 of
/// [eprint 2023/771](https://eprint.iacr.org/2023/771).
fn ensure_accumulation_fits<D: poulpy_hal::layouts::Data>(op: &'static str, dst: &CKKSCiphertext<D>, n: usize) -> Result<()> {
    let base2k: usize = dst.base2k().as_usize();
    ensure!(base2k < 64, "{op}: unsupported base2k={base2k}");
    ensure!(
        n <= (1usize << (63 - base2k)),
        "{op}: {n} terms risks i64 overflow at base2k={base2k}",
    );
    Ok(())
}

/// Shared accumulation loop: `dst += Σ_{i≥1} mul_term(i)`, finished with a
/// single trailing normalize. `dst` must already hold the first product.
fn accumulate_unnormalized<BE, D, F>(
    module: &Module<BE>,
    dst: &mut CKKSCiphertext<D>,
    n: usize,
    scratch: &mut Scratch<BE>,
    mut mul_term_into_tmp: F,
) -> Result<()>
where
    BE: Backend + CKKSImpl<BE>,
    D: DataMut,
    Module<BE>: GLWEAdd + GLWEShift<BE> + GLWENormalize<BE> + CKKSAddOpsWithoutNormalization<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    F: FnMut(&mut CKKSCiphertext<&mut [u8]>, usize, &mut Scratch<BE>) -> Result<()>,
{
    if n <= 1 {
        return Ok(());
    }
    let layout = dst.glwe_layout();
    let (tmp_glwe, scratch_r) = scratch.take_glwe(&layout);
    let mut tmp: CKKSCiphertext<&mut [u8]> = CKKSCiphertext::from_inner(tmp_glwe, CKKSMeta::default());
    for i in 1..n {
        mul_term_into_tmp(&mut tmp, i, scratch_r)?;
        unsafe {
            module.ckks_add_inplace_without_normalization(dst, &tmp, scratch_r)?;
        }
    }
    module.glwe_normalize_inplace(dst, scratch_r);
    Ok(())
}

impl<BE: Backend + CKKSImpl<BE>> CKKSDotProductOps<BE> for Module<BE> {
    fn ckks_dot_product_ct_tmp_bytes<R, T>(&self, n: usize, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWEShift<BE> + GLWETensoring<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
    {
        let mul_scratch: usize = self.ckks_mul_tmp_bytes(res, tsk);
        if n <= 1 {
            return mul_scratch;
        }
        let ct_bytes: usize = GLWE::<Vec<u8>>::bytes_of_from_infos(res);
        // Non-uniform `log_decimal` fallback: one tmp ciphertext + per-pair mul or add.
        let fallback: usize = ct_bytes + mul_scratch.max(self.ckks_add_tmp_bytes());
        // Fast path (worst case: both sides unaligned): `n` rescale buffers per
        // side + one tensor accumulator + inner.
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

    fn ckks_dot_product_pt_vec_znx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_znx_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_dot_product_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_rnx_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_dot_product_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_const_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_dot_product_ct<D: DataRef, E: DataRef>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSCiphertext<E>],
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + GLWETensoring<BE>
            + VecZnxAddAssign
            + CKKSAddOpsWithoutNormalization<BE>
            + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        check_lengths("ckks_dot_product_ct", a.len(), b.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_ct", dst, n)?;

        if n == 1 {
            return self.ckks_mul(dst, a[0], b[0], tsk, scratch);
        }

        // The deferred-relinearize path sums per-pair tensors termwise in ℤ,
        // so every pair must produce a tensor at the same semantic scale.
        // Heterogeneous `log_hom_rem` is handled by rescaling each side to
        // its per-side minimum up front (precision-neutral vs. the per-pair
        // `ckks_mul + ckks_add_inplace` baseline); mismatched `log_decimal`
        // within a side has no such equalization and falls back.
        let a_min_lhr: usize = a.iter().map(|c| c.log_hom_rem()).min().unwrap();
        let b_min_lhr: usize = b.iter().map(|c| c.log_hom_rem()).min().unwrap();
        let a_aligned: bool = a
            .iter()
            .all(|c| c.log_hom_rem() == a_min_lhr && c.log_decimal() == a[0].log_decimal());
        let b_aligned: bool = b
            .iter()
            .all(|c| c.log_hom_rem() == b_min_lhr && c.log_decimal() == b[0].log_decimal());
        let uniform_ld =
            a.iter().all(|c| c.log_decimal() == a[0].log_decimal()) && b.iter().all(|c| c.log_decimal() == b[0].log_decimal());

        if !uniform_ld {
            self.ckks_mul(dst, a[0], b[0], tsk, scratch)?;
            return accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| self.ckks_mul(tmp, a[i], b[i], tsk, s));
        }

        let a_ld: usize = a[0].log_decimal();
        let b_ld: usize = b[0].log_decimal();
        let a_target_eff_k: usize = a_min_lhr + a_ld;
        let b_target_eff_k: usize = b_min_lhr + b_ld;

        let a_layout = GLWELayout {
            n: dst.n(),
            base2k: dst.base2k(),
            k: TorusPrecision(a_target_eff_k as u32),
            rank: dst.rank(),
        };
        let b_layout = GLWELayout {
            n: dst.n(),
            base2k: dst.base2k(),
            k: TorusPrecision(b_target_eff_k as u32),
            rank: dst.rank(),
        };

        let (a_buf_raw, scratch_aa) = if a_aligned {
            (Vec::new(), &mut *scratch)
        } else {
            scratch.take_glwe_slice(n, &a_layout)
        };
        let (b_buf_raw, scratch_ab) = if b_aligned {
            (Vec::new(), scratch_aa)
        } else {
            scratch_aa.take_glwe_slice(n, &b_layout)
        };

        let mut a_buf: Vec<CKKSCiphertext<&mut [u8]>> = a_buf_raw
            .into_iter()
            .map(|g| CKKSCiphertext::from_inner(g, CKKSMeta::default()))
            .collect();
        let mut b_buf: Vec<CKKSCiphertext<&mut [u8]>> = b_buf_raw
            .into_iter()
            .map(|g| CKKSCiphertext::from_inner(g, CKKSMeta::default()))
            .collect();

        if !a_aligned {
            for (i, ai) in a.iter().enumerate() {
                let shift = ai.log_hom_rem() - a_min_lhr;
                self.ckks_rescale(&mut a_buf[i], shift, *ai, scratch_ab)?;
            }
        }
        if !b_aligned {
            for (i, bi) in b.iter().enumerate() {
                let shift = bi.log_hom_rem() - b_min_lhr;
                self.ckks_rescale(&mut b_buf[i], shift, *bi, scratch_ab)?;
            }
        }

        let dst_max_k: usize = dst.max_k().as_usize();
        let res_log_decimal: usize = a_ld.min(b_ld);
        let lhr0: usize = checked_mul_ct_log_hom_rem("dot_product_ct", a_min_lhr, b_min_lhr, a_ld, b_ld)?;
        let res_offset: usize = (lhr0 + res_log_decimal).saturating_sub(dst_max_k);
        let res_log_hom_rem: usize = checked_log_hom_rem_sub("dot_product_ct", lhr0, res_offset)?;
        let cnv_offset: usize = a_target_eff_k.max(b_target_eff_k) + res_offset;

        let tensor_max_k: usize = a_target_eff_k.max(b_target_eff_k);
        let tensor_layout = GLWELayout {
            n: dst.n(),
            base2k: dst.base2k(),
            k: TorusPrecision(tensor_max_k as u32),
            rank: dst.rank(),
        };

        let (mut acc_tensor, scratch_t2) = scratch_ab.take_glwe_tensor(&tensor_layout);

        let a0_ref: GLWE<&[u8]> = if a_aligned { a[0].to_ref() } else { a_buf[0].to_ref() };
        let b0_ref: GLWE<&[u8]> = if b_aligned { b[0].to_ref() } else { b_buf[0].to_ref() };
        self.glwe_tensor_apply(
            cnv_offset,
            &mut acc_tensor,
            &a0_ref,
            a_target_eff_k,
            &b0_ref,
            b_target_eff_k,
            scratch_t2,
        );

        for i in 1..n {
            let ai_ref: GLWE<&[u8]> = if a_aligned { a[i].to_ref() } else { a_buf[i].to_ref() };
            let bi_ref: GLWE<&[u8]> = if b_aligned { b[i].to_ref() } else { b_buf[i].to_ref() };
            self.glwe_tensor_apply_add_assign(
                cnv_offset,
                &mut acc_tensor,
                &ai_ref,
                a_target_eff_k,
                &bi_ref,
                b_target_eff_k,
                scratch_t2,
            );
        }

        self.glwe_tensor_relinearize(&mut dst.to_mut(), &acc_tensor, tsk, tsk.size(), scratch_t2);
        dst.meta.log_hom_rem = res_log_hom_rem;
        dst.meta.log_decimal = res_log_decimal;
        Ok(())
    }

    fn ckks_dot_product_pt_vec_znx<D: DataRef, E: DataRef>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextVecZnx<E>],
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd
            + GLWEMulPlain<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + VecZnxRshAddInto<BE>
            + CKKSAddOpsWithoutNormalization<BE>
            + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        check_lengths("ckks_dot_product_pt_vec_znx", a.len(), b.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_pt_vec_znx", dst, n)?;
        self.ckks_mul_pt_vec_znx(dst, a[0], b[0], scratch)?;
        accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| {
            self.ckks_mul_pt_vec_znx(tmp, a[i], b[i], s)
        })
    }

    fn ckks_dot_product_pt_vec_rnx<D: DataRef, F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextVecRnx<F>],
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN
            + GLWEAdd
            + GLWEMulPlain<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + VecZnxRshAddInto<BE>
            + CKKSAddOpsWithoutNormalization<BE>
            + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        check_lengths("ckks_dot_product_pt_vec_rnx", a.len(), b.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_pt_vec_rnx", dst, n)?;
        self.ckks_mul_pt_vec_rnx(dst, a[0], b[0], prec, scratch)?;
        accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| {
            self.ckks_mul_pt_vec_rnx(tmp, a[i], b[i], prec, s)
        })
    }

    fn ckks_dot_product_const_znx<D: DataRef>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextCstZnx],
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + CKKSAddOpsWithoutNormalization<BE>
            + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        check_lengths("ckks_dot_product_const_znx", a.len(), b.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_const_znx", dst, n)?;
        self.ckks_mul_pt_const_znx(dst, a[0], b[0], scratch)?;
        accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| {
            self.ckks_mul_pt_const_znx(tmp, a[i], b[i], s)
        })
    }

    fn ckks_dot_product_const_rnx<D: DataRef, F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextCstRnx<F>],
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + CKKSAddOpsWithoutNormalization<BE>
            + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        check_lengths("ckks_dot_product_const_rnx", a.len(), b.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_const_rnx", dst, n)?;
        self.ckks_mul_pt_const_rnx(dst, a[0], b[0], prec, scratch)?;
        accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| {
            self.ckks_mul_pt_const_rnx(tmp, a[i], b[i], prec, s)
        })
    }
}
