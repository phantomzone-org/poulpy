//! CKKS inner product `dst = Σ aᵢ · bᵢ`.

use anyhow::{Result, bail};
use poulpy_core::{
    GLWEAdd, GLWEMulConst, GLWEMulPlain, GLWENormalize, GLWERotate, GLWEShift, GLWETensoring, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWETensor, GLWETensorKeyPrepared, GLWEToMut, GLWEToRef, LWEInfos,
        TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxAddAssign, VecZnxRshAddInto, VecZnxZero},
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
    leveled::operations::{
        add::{CKKSAddOps, CKKSAddOpsWithoutNormalization},
        mul::CKKSMulOps,
    },
    oep::CKKSImpl,
};

pub trait CKKSDotProductOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_dot_product_ct_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
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
            + VecZnxZero
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
fn assert_accumulation_fits<D: poulpy_hal::layouts::Data>(op: &'static str, dst: &CKKSCiphertext<D>, n: usize) {
    let base2k: usize = dst.base2k().as_usize();
    debug_assert!(
        base2k < 64 && n <= (1usize << (63 - base2k)),
        "{op}: {n} terms risks i64 overflow at base2k={base2k}",
    );
}

impl<BE: Backend + CKKSImpl<BE>> CKKSDotProductOps<BE> for Module<BE> {
    fn ckks_dot_product_ct_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWEShift<BE> + GLWETensoring<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
    {
        // Deferred-relinearization path: two GLWETensor slots (accumulator +
        // per-pair scratch) plus the heavier of a tensor apply and a
        // relinearize.
        let tensor_layout = GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: TorusPrecision(res.max_k().as_u32()),
            rank: res.rank(),
        };
        let tensor_bytes: usize = GLWETensor::bytes_of_from_infos(&tensor_layout);
        let inner: usize = self
            .glwe_tensor_apply_tmp_bytes(&tensor_layout, res, res)
            .max(self.glwe_tensor_relinearize_tmp_bytes(res, &tensor_layout, tsk));
        2 * tensor_bytes + inner
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
            + VecZnxZero
            + CKKSAddOpsWithoutNormalization<BE>
            + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        check_lengths("ckks_dot_product_ct", a.len(), b.len())?;
        let n: usize = a.len();
        assert_accumulation_fits("ckks_dot_product_ct", dst, n);

        // Single-term fast path: one mul is cheaper than tensor + relinearize.
        if n == 1 {
            return self.ckks_mul(dst, a[0], b[0], tsk, scratch);
        }

        // Deferred relinearization requires every pair to produce a tensor
        // at the same semantic scale. The shift logic in `ckks_add_inplace`
        // compensates for mismatched `log_hom_rem` in the per-pair loop; we
        // don't have that in the tensor domain, so if inputs are not all
        // aligned we fall back to the correct-but-slower path.
        let aligned = a.iter().zip(b.iter()).all(|(ai, bi)| {
            ai.log_hom_rem() == a[0].log_hom_rem()
                && bi.log_hom_rem() == b[0].log_hom_rem()
                && ai.log_decimal() == a[0].log_decimal()
                && bi.log_decimal() == b[0].log_decimal()
        });

        if !aligned {
            self.ckks_mul(dst, a[0], b[0], tsk, scratch)?;
            let (tmp_glwe, scratch_r) = scratch.take_glwe(&dst.glwe_layout());
            let mut tmp: CKKSCiphertext<&mut [u8]> = CKKSCiphertext::from_inner(tmp_glwe, CKKSMeta::default());
            for i in 1..n {
                self.ckks_mul(&mut tmp, a[i], b[i], tsk, scratch_r)?;
                // SAFETY: the trailing glwe_normalize_inplace restores
                // K-normalization.
                unsafe {
                    self.ckks_add_inplace_without_normalization(dst, &tmp, scratch_r)?;
                }
            }
            self.glwe_normalize_inplace(dst, scratch_r);
            return Ok(());
        }

        // Aligned path: compute each term's tensor product, accumulate the
        // tensors in ℤ, then relinearize the accumulator once. Saves `n − 1`
        // relinearizations compared to the per-pair `ckks_mul` loop.
        let dst_max_k: usize = dst.max_k().as_usize();
        let lhr = checked_mul_ct_log_hom_rem(
            "dot_product_ct",
            a[0].log_hom_rem(),
            b[0].log_hom_rem(),
            a[0].log_decimal(),
            b[0].log_decimal(),
        )?;
        let res_log_decimal: usize = a[0].log_decimal().min(b[0].log_decimal());
        let res_offset: usize = (lhr + res_log_decimal).saturating_sub(dst_max_k);
        let res_log_hom_rem: usize = checked_log_hom_rem_sub("dot_product_ct", lhr, res_offset)?;
        let cnv_offset: usize = a[0].effective_k().max(b[0].effective_k()) + res_offset;

        let tensor_max_k: usize = a[0].max_k().as_usize().max(b[0].max_k().as_usize());
        let tensor_layout = GLWELayout {
            n: dst.n(),
            base2k: dst.base2k(),
            k: TorusPrecision(tensor_max_k as u32),
            rank: dst.rank(),
        };

        let (mut acc_tensor, scratch_a) = scratch.take_glwe_tensor(&tensor_layout);
        let (mut tmp_tensor, scratch_b) = scratch_a.take_glwe_tensor(&tensor_layout);

        let pairs: usize = ((acc_tensor.rank().as_usize() + 1) * (acc_tensor.rank().as_usize() + 2)) / 2;
        for i in 0..pairs {
            self.vec_znx_zero(acc_tensor.data_mut(), i);
        }

        for (ai, bi) in a.iter().zip(b.iter()) {
            self.glwe_tensor_apply(
                cnv_offset,
                &mut tmp_tensor,
                &ai.to_ref(),
                ai.effective_k(),
                &bi.to_ref(),
                bi.effective_k(),
                scratch_b,
            );
            for col in 0..pairs {
                self.vec_znx_add_assign(acc_tensor.data_mut(), col, tmp_tensor.data(), col);
            }
        }

        self.glwe_tensor_relinearize(&mut dst.to_mut(), &acc_tensor, tsk, tsk.size(), scratch_b);
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
        assert_accumulation_fits("ckks_dot_product_pt_vec_znx", dst, n);
        self.ckks_mul_pt_vec_znx(dst, a[0], b[0], scratch)?;
        if n == 1 {
            return Ok(());
        }
        let (tmp_glwe, scratch_r) = scratch.take_glwe(&dst.glwe_layout());
        let mut tmp: CKKSCiphertext<&mut [u8]> = CKKSCiphertext::from_inner(tmp_glwe, CKKSMeta::default());
        for i in 1..n {
            self.ckks_mul_pt_vec_znx(&mut tmp, a[i], b[i], scratch_r)?;
            unsafe {
                self.ckks_add_inplace_without_normalization(dst, &tmp, scratch_r)?;
            }
        }
        self.glwe_normalize_inplace(dst, scratch_r);
        Ok(())
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
        assert_accumulation_fits("ckks_dot_product_pt_vec_rnx", dst, n);
        self.ckks_mul_pt_vec_rnx(dst, a[0], b[0], prec, scratch)?;
        if n == 1 {
            return Ok(());
        }
        let (tmp_glwe, scratch_r) = scratch.take_glwe(&dst.glwe_layout());
        let mut tmp: CKKSCiphertext<&mut [u8]> = CKKSCiphertext::from_inner(tmp_glwe, CKKSMeta::default());
        for i in 1..n {
            self.ckks_mul_pt_vec_rnx(&mut tmp, a[i], b[i], prec, scratch_r)?;
            unsafe {
                self.ckks_add_inplace_without_normalization(dst, &tmp, scratch_r)?;
            }
        }
        self.glwe_normalize_inplace(dst, scratch_r);
        Ok(())
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
        assert_accumulation_fits("ckks_dot_product_const_znx", dst, n);
        self.ckks_mul_pt_const_znx(dst, a[0], b[0], scratch)?;
        if n == 1 {
            return Ok(());
        }
        let (tmp_glwe, scratch_r) = scratch.take_glwe(&dst.glwe_layout());
        let mut tmp: CKKSCiphertext<&mut [u8]> = CKKSCiphertext::from_inner(tmp_glwe, CKKSMeta::default());
        for i in 1..n {
            self.ckks_mul_pt_const_znx(&mut tmp, a[i], b[i], scratch_r)?;
            unsafe {
                self.ckks_add_inplace_without_normalization(dst, &tmp, scratch_r)?;
            }
        }
        self.glwe_normalize_inplace(dst, scratch_r);
        Ok(())
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
        assert_accumulation_fits("ckks_dot_product_const_rnx", dst, n);
        self.ckks_mul_pt_const_rnx(dst, a[0], b[0], prec, scratch)?;
        if n == 1 {
            return Ok(());
        }
        let (tmp_glwe, scratch_r) = scratch.take_glwe(&dst.glwe_layout());
        let mut tmp: CKKSCiphertext<&mut [u8]> = CKKSCiphertext::from_inner(tmp_glwe, CKKSMeta::default());
        for i in 1..n {
            self.ckks_mul_pt_const_rnx(&mut tmp, a[i], b[i], prec, scratch_r)?;
            unsafe {
                self.ckks_add_inplace_without_normalization(dst, &tmp, scratch_r)?;
            }
        }
        self.glwe_normalize_inplace(dst, scratch_r);
        Ok(())
    }
}
