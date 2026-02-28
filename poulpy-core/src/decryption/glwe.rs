use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, ScratchTakeBasic, SvpApplyDftToDftInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace,
        VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyConsume, VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, DataRef, DataViewMut, Module, Scratch},
};

use crate::layouts::{
    GLWE, GLWEInfos, GLWEPlaintextToMut, GLWESecretPrepared, GLWEToRef, LWEInfos, SetGLWEInfos, prepared::GLWESecretPreparedToRef,
};

impl GLWE<Vec<u8>> {
    /// Returns the number of scratch bytes required by [`GLWE::decrypt`].
    pub fn decrypt_tmp_bytes<A, M, BE: Backend>(module: &M, a_infos: &A) -> usize
    where
        A: GLWEInfos,
        M: GLWEDecrypt<BE>,
    {
        module.glwe_decrypt_tmp_bytes(a_infos)
    }
}

impl<DataSelf: DataRef> GLWE<DataSelf> {
    /// Decrypts this GLWE ciphertext into a plaintext using the prepared secret key.
    ///
    /// Computes `pt = body + mask * secret`, where `body` is the first column
    /// of the ciphertext and `mask` comprises the remaining columns. The result
    /// is then normalized into the plaintext decomposition basis.
    ///
    /// The plaintext precision `k` is set to the minimum of the plaintext and
    /// ciphertext precisions.
    pub fn decrypt<P, S, M, BE: Backend>(&self, module: &M, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        P: GLWEPlaintextToMut + GLWEInfos + SetGLWEInfos,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        M: GLWEDecrypt<BE>,
        Scratch<BE>: ScratchTakeBasic,
    {
        module.glwe_decrypt(self, pt, sk, scratch);
    }
}

pub trait GLWEDecrypt<BE: Backend> {
    fn glwe_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_decrypt<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        R: GLWEToRef + GLWEInfos,
        P: GLWEPlaintextToMut + GLWEInfos + SetGLWEInfos,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos;
}

impl<BE: Backend> GLWEDecrypt<BE> for Module<BE>
where
    Self: ModuleN
        + VecZnxDftBytesOf
        + VecZnxNormalizeTmpBytes
        + VecZnxBigBytesOf
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftInplace<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigAddInplace<BE>
        + VecZnxBigAddSmallInplace<BE>
        + VecZnxBigNormalize<BE>,
    Scratch<BE>: ScratchTakeBasic + ScratchAvailable,
{
    fn glwe_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let size: usize = infos.size();
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = self.bytes_of_vec_znx_big(1, size);
        let lvl_1: usize = self.bytes_of_vec_znx_dft(1, size).max(self.vec_znx_normalize_tmp_bytes());

        lvl_0 + lvl_1
    }

    fn glwe_decrypt<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        R: GLWEToRef + GLWEInfos,
        P: GLWEPlaintextToMut + GLWEInfos + SetGLWEInfos,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
    {
        let res: &GLWE<&[u8]> = &res.to_ref();
        let sk: &GLWESecretPrepared<&[u8], BE> = &sk.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.rank(), sk.rank()); //NOTE: res.rank() != res.to_ref().rank() if res is of type GLWETensor
            assert_eq!(res.n(), sk.n());
            assert_eq!(pt.n(), sk.n());
        }
        assert!(
            scratch.available() >= self.glwe_decrypt_tmp_bytes(res),
            "scratch.available(): {} < GLWEDecrypt::glwe_decrypt_tmp_bytes: {}",
            scratch.available(),
            self.glwe_decrypt_tmp_bytes(res)
        );

        let cols: usize = (res.rank() + 1).into();

        let (mut c0_big, scratch_1) = scratch.take_vec_znx_big(self, 1, res.size()); // TODO optimize size when pt << ct
        c0_big.data_mut().fill(0);

        (1..cols).for_each(|i| {
            // ci_dft = DFT(a[i]) * DFT(s[i])
            let (mut ci_dft, _) = scratch_1.take_vec_znx_dft(self, 1, res.size()); // TODO optimize size when pt << ct
            self.vec_znx_dft_apply(1, 0, &mut ci_dft, 0, res.data(), i);
            self.svp_apply_dft_to_dft_inplace(&mut ci_dft, 0, &sk.data, i - 1);
            let ci_big = self.vec_znx_idft_apply_consume(ci_dft);

            // c0_big += a[i] * s[i]
            self.vec_znx_big_add_inplace(&mut c0_big, 0, &ci_big, 0);
        });

        // c0_big = (a * s) + (-a * s + m + e) = BIG(m + e)
        self.vec_znx_big_add_small_inplace(&mut c0_big, 0, res.data(), 0);

        let pt_base2k: usize = pt.base2k().into();

        // pt = norm(BIG(m + e))
        self.vec_znx_big_normalize(
            pt.to_mut().data_mut(),
            pt_base2k,
            0,
            0,
            &c0_big,
            res.base2k().into(),
            0,
            scratch_1,
        );

        pt.set_k(pt.k().min(res.k()));
    }
}
