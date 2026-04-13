use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, ScratchTakeBasic, SvpApplyDftToDftInplace, VecZnxBigAddAssign, VecZnxBigAddSmallAssign,
        VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyConsume, VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, DataRef, DataViewMut, Module, Scratch},
};

pub use crate::api::GLWEDecrypt;
use crate::layouts::{
    GLWE, GLWEInfos, GLWEPlaintextToMut, GLWESecretPrepared, GLWEToRef, LWEInfos, SetGLWEInfos, prepared::GLWESecretPreparedToRef,
};

impl<M> GLWE<Vec<u8>, M> {
    /// Returns the number of scratch bytes required by [`GLWE::decrypt`].
    pub fn decrypt_tmp_bytes<A, Mod, BE: Backend>(module: &Mod, a_infos: &A) -> usize
    where
        A: GLWEInfos,
        Mod: GLWEDecrypt<BE>,
    {
        module.glwe_decrypt_tmp_bytes(a_infos)
    }
}

impl<DataSelf: DataRef, M> GLWE<DataSelf, M> {
    /// Decrypts this GLWE ciphertext into a plaintext using the prepared secret key.
    ///
    /// Computes `pt = body + mask * secret`, where `body` is the first column
    /// of the ciphertext and `mask` comprises the remaining columns. The result
    /// is then normalized into the plaintext decomposition basis.
    ///
    /// The plaintext precision `k` is set to the minimum of the plaintext and
    /// ciphertext precisions.
    pub fn decrypt<P, S, Mod, BE: Backend>(&self, module: &Mod, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        P: GLWEPlaintextToMut + GLWEInfos + SetGLWEInfos,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        Mod: GLWEDecrypt<BE>,
        Scratch<BE>: ScratchTakeBasic,
    {
        module.glwe_decrypt(self, pt, sk, scratch);
    }
}

pub(crate) trait GLWEDecryptDefault<BE: Backend>:
    Sized
    + ModuleN
    + VecZnxDftBytesOf
    + VecZnxNormalizeTmpBytes
    + VecZnxBigBytesOf
    + VecZnxDftApply<BE>
    + SvpApplyDftToDftInplace<BE>
    + VecZnxIdftApplyConsume<BE>
    + VecZnxBigAddAssign<BE>
    + VecZnxBigAddSmallAssign<BE>
    + VecZnxBigNormalize<BE>
where
    Scratch<BE>: ScratchTakeBasic + ScratchAvailable,
{
    fn glwe_decrypt_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let size: usize = infos.size();
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = self.bytes_of_vec_znx_big(1, size);
        let lvl_1: usize = self.bytes_of_vec_znx_dft(1, size).max(self.vec_znx_normalize_tmp_bytes());

        lvl_0 + lvl_1
    }

    fn glwe_decrypt_default<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
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
            scratch.available() >= self.glwe_decrypt_tmp_bytes_default(res),
            "scratch.available(): {} < GLWEDecrypt::glwe_decrypt_tmp_bytes: {}",
            scratch.available(),
            self.glwe_decrypt_tmp_bytes_default(res)
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
            self.vec_znx_big_add_assign(&mut c0_big, 0, &ci_big, 0);
        });

        // c0_big = (a * s) + (-a * s + m + e) = BIG(m + e)
        self.vec_znx_big_add_small_assign(&mut c0_big, 0, res.data(), 0);

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
    }
}

impl<BE: Backend> GLWEDecryptDefault<BE> for Module<BE>
where
    Self: ModuleN
        + VecZnxDftBytesOf
        + VecZnxNormalizeTmpBytes
        + VecZnxBigBytesOf
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftInplace<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>,
    Scratch<BE>: ScratchTakeBasic + ScratchAvailable,
{
}
