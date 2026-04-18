use poulpy_core::{
    ScratchTakeCore,
    layouts::{GLWEPlaintext, LWEInfos},
};
use poulpy_hal::{
    api::{VecZnxLsh, VecZnxLshTmpBytes, VecZnxRsh, VecZnxRshAddInto, VecZnxRshSub, VecZnxRshTmpBytes},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{
    CKKSInfos, ensure_base2k_match, ensure_plaintext_alignment,
    layouts::{CKKSCiphertext, CKKSPlaintextVecZnx},
};
use anyhow::Result;

pub trait CKKSPlaintextZnxOps<BE: Backend> {
    /// Returns the scratch size, in bytes, required by [`Self::ckks_add_pt_vec_znx`].
    ///
    /// The returned size depends only on the backend/module configuration.
    /// It does not inspect any ciphertext or plaintext inputs.
    fn ckks_add_pt_znx_tmp_bytes(&self) -> usize
    where
        Self: VecZnxRshTmpBytes;

    /// Adds a compact CKKS plaintext vector into a ciphertext in place.
    ///
    /// Inputs:
    /// - `dst`: ciphertext updated in place
    /// - `src`: plaintext vector to add
    /// - `scratch`: temporary workspace
    ///
    /// Output:
    /// - Returns `Ok(())` and mutates `dst` on success
    ///
    /// Behavior:
    /// - `src` is interpreted at its own `log_decimal`
    /// - the plaintext is right-shifted as needed so its semantic position
    ///   matches `dst`
    /// - ciphertext metadata is not changed
    ///
    /// Errors:
    /// - `PlaintextBase2KMismatch` if `dst` and `src` use different `base2k`
    /// - `PlaintextAlignmentImpossible` if `src` asks for more semantic width
    ///   than can be aligned into `dst` at its current `log_hom_rem`
    fn ckks_add_pt_vec_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxRshAddInto<BE>;

    /// Returns the scratch size, in bytes, required by [`Self::ckks_sub_pt_vec_znx`].
    ///
    /// The returned size depends only on the backend/module configuration.
    fn ckks_sub_pt_znx_tmp_bytes(&self) -> usize
    where
        Self: VecZnxRshTmpBytes;

    /// Subtracts a compact CKKS plaintext vector from a ciphertext in place.
    ///
    /// Inputs:
    /// - `dst`: ciphertext updated in place
    /// - `src`: plaintext vector to subtract
    /// - `scratch`: temporary workspace
    ///
    /// Output:
    /// - Returns `Ok(())` and mutates `dst` on success
    ///
    /// Behavior:
    /// - `src` is aligned to `dst` using a right shift before subtraction
    /// - ciphertext metadata is not changed
    ///
    /// Errors:
    /// - `PlaintextBase2KMismatch` if `dst` and `src` use different `base2k`
    /// - `PlaintextAlignmentImpossible` if `src` cannot be aligned into `dst`
    fn ckks_sub_pt_vec_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxRshSub<BE>;

    /// Returns the scratch size, in bytes, required by [`Self::ckks_extract_pt_znx`].
    ///
    /// Extraction may need either a left or right shift depending on the
    /// destination plaintext layout, so this returns the maximum of both
    /// backend scratch requirements.
    fn ckks_extract_pt_znx_tmp_bytes(&self) -> usize
    where
        Self: VecZnxLshTmpBytes + VecZnxRshTmpBytes;

    /// Extracts a decrypted GLWE plaintext into a CKKS plaintext vector with a
    /// caller-chosen destination precision.
    ///
    /// Inputs:
    /// - `dst`: destination CKKS plaintext buffer and metadata
    /// - `src`: raw decrypted GLWE plaintext limbs
    /// - `src_meta`: source CKKS metadata, typically from the ciphertext that
    ///   produced `src`
    /// - `scratch`: temporary workspace
    ///
    /// Output:
    /// - Returns `Ok(())` and fills `dst` on success
    ///
    /// Behavior:
    /// - the destination metadata is treated as a contract and is not mutated
    /// - let `available = src_meta.log_hom_rem() + dst.log_decimal()`
    /// - if `available < dst.max_k()`, the extracted value is right-shifted so
    ///   the semantic plaintext lands in the destination limbs
    /// - if `available > dst.max_k()`, the extracted value is left-shifted so
    ///   the destination remains aligned while truncating excess upper
    ///   homomorphic capacity
    /// - if `available == dst.max_k()`, extraction copies with no net shift
    ///
    /// Errors:
    /// - `PlaintextBase2KMismatch` if `src` and `dst` use different `base2k`
    /// - `PlaintextAlignmentImpossible` if `dst.effective_k()` exceeds the
    ///   semantic width available from `src_meta`
    fn ckks_extract_pt_znx<S>(
        &self,
        dst: &mut CKKSPlaintextVecZnx<impl DataMut>,
        src: &GLWEPlaintext<impl DataRef>,
        src_meta: &S,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        S: CKKSInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxLsh<BE> + VecZnxRsh<BE>;
}

impl<BE: Backend> CKKSPlaintextZnxOps<BE> for Module<BE> {
    fn ckks_add_pt_znx_tmp_bytes(&self) -> usize
    where
        Self: VecZnxRshTmpBytes,
    {
        self.vec_znx_rsh_tmp_bytes()
    }

    fn ckks_add_pt_vec_znx(
        &self,
        ct: &mut CKKSCiphertext<impl DataMut>,
        pt: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxRshAddInto<BE>,
    {
        ensure_base2k_match("ckks_add_pt_vec_znx", ct.base2k().as_usize(), pt.base2k().as_usize())?;
        let offset = ensure_plaintext_alignment(
            "ckks_add_pt_vec_znx",
            ct.log_hom_rem(),
            pt.log_decimal(),
            pt.max_k().as_usize(),
        )?;
        self.vec_znx_rsh_add_into(ct.base2k().as_usize(), offset, ct.data_mut(), 0, pt.data(), 0, scratch);
        Ok(())
    }

    fn ckks_sub_pt_znx_tmp_bytes(&self) -> usize
    where
        Self: VecZnxRshTmpBytes,
    {
        self.vec_znx_rsh_tmp_bytes()
    }

    fn ckks_extract_pt_znx_tmp_bytes(&self) -> usize
    where
        Self: VecZnxLshTmpBytes + VecZnxRshTmpBytes,
    {
        self.vec_znx_rsh_tmp_bytes().max(self.vec_znx_lsh_tmp_bytes())
    }

    fn ckks_extract_pt_znx<S>(
        &self,
        dst: &mut CKKSPlaintextVecZnx<impl DataMut>,
        src: &GLWEPlaintext<impl DataRef>,
        src_meta: &S,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        S: CKKSInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxLsh<BE> + VecZnxRsh<BE>,
    {
        ensure_base2k_match("ckks_extract_pt_znx", src.base2k().as_usize(), dst.base2k().as_usize())?;
        let available = src_meta.log_hom_rem() + dst.log_decimal();
        if available < dst.effective_k() {
            return Err(crate::CKKSCompositionError::PlaintextAlignmentImpossible {
                op: "ckks_extract_pt_znx",
                ct_log_hom_rem: src_meta.log_hom_rem(),
                pt_log_decimal: dst.log_decimal(),
                pt_max_k: dst.max_k().as_usize(),
            }
            .into());
        }
        let dst_k = dst.max_k().as_usize();
        if available < dst_k {
            self.vec_znx_rsh(
                dst.base2k().into(),
                dst_k - available,
                dst.data_mut(),
                0,
                src.data(),
                0,
                scratch,
            );
        } else if available > dst_k {
            self.vec_znx_lsh(
                dst.base2k().into(),
                available - dst_k,
                dst.data_mut(),
                0,
                src.data(),
                0,
                scratch,
            );
        } else {
            self.vec_znx_rsh(dst.base2k().into(), 0, dst.data_mut(), 0, src.data(), 0, scratch);
        }
        Ok(())
    }

    fn ckks_sub_pt_vec_znx(
        &self,
        ct: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxRshSub<BE>,
    {
        ensure_base2k_match("ckks_sub_pt_vec_znx", ct.base2k().as_usize(), pt_znx.base2k().as_usize())?;
        let offset = ensure_plaintext_alignment(
            "ckks_sub_pt_vec_znx",
            ct.log_hom_rem(),
            pt_znx.log_decimal(),
            pt_znx.max_k().as_usize(),
        )?;
        self.vec_znx_rsh_sub(ct.base2k().as_usize(), offset, ct.data_mut(), 0, pt_znx.data(), 0, scratch);
        Ok(())
    }
}
