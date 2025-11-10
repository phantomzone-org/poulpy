use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalizeInplace, VecZnxSubInplace},
    layouts::{Backend, DataRef, Module, Scratch, ScratchOwned, Stats},
};

use crate::{
    ScratchTakeCore,
    decryption::GLWEDecrypt,
    layouts::{GLWE, GLWEPlaintext, GLWEPlaintextToRef, GLWEToRef, LWEInfos, prepared::GLWESecretPreparedToRef},
};

impl<D: DataRef> GLWE<D> {
    pub fn noise<M, S, P, BE: Backend>(&self, module: &M, sk_prepared: &S, pt_want: &P, scratch: &mut Scratch<BE>) -> Stats
    where
        M: GLWENoise<BE>,
        S: GLWESecretPreparedToRef<BE>,
        P: GLWEPlaintextToRef,
    {
        module.glwe_noise(self, sk_prepared, pt_want, scratch)
    }

    pub fn assert_noise<M, BE: Backend, S, P>(&self, module: &M, sk_prepared: &S, pt_want: &P, max_noise: f64)
    where
        S: GLWESecretPreparedToRef<BE>,
        P: GLWEPlaintextToRef,
        M: GLWENoise<BE>,
    {
        module.glwe_assert_noise(self, sk_prepared, pt_want, max_noise);
    }
}

pub trait GLWENoise<BE: Backend> {
    fn glwe_noise<R, S, P>(&self, res: &R, sk_prepared: &S, pt_want: &P, scratch: &mut Scratch<BE>) -> Stats
    where
        R: GLWEToRef,
        S: GLWESecretPreparedToRef<BE>,
        P: GLWEPlaintextToRef;

    fn glwe_assert_noise<R, S, P>(&self, res: &R, sk_prepared: &S, pt_want: &P, max_noise: f64)
    where
        R: GLWEToRef,
        S: GLWESecretPreparedToRef<BE>,
        P: GLWEPlaintextToRef;
}

impl<BE: Backend> GLWENoise<BE> for Module<BE>
where
    Module<BE>: GLWEDecrypt<BE> + VecZnxSubInplace + VecZnxNormalizeInplace<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_noise<R, S, P>(&self, res: &R, sk_prepared: &S, pt_want: &P, scratch: &mut Scratch<BE>) -> Stats
    where
        R: GLWEToRef,
        S: GLWESecretPreparedToRef<BE>,
        P: GLWEPlaintextToRef,
    {
        let res_ref: &GLWE<&[u8]> = &res.to_ref();

        let pt_want: &GLWEPlaintext<&[u8]> = &pt_want.to_ref();

        let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(res_ref);
        self.glwe_decrypt(res, &mut pt_have, sk_prepared, scratch);
        self.vec_znx_sub_inplace(&mut pt_have.data, 0, &pt_want.data, 0);
        self.vec_znx_normalize_inplace(res_ref.base2k().into(), &mut pt_have.data, 0, scratch);
        pt_have.data.stats(res_ref.base2k().into(), 0)
    }

    fn glwe_assert_noise<R, S, P>(&self, res: &R, sk_prepared: &S, pt_want: &P, max_noise: f64)
    where
        R: GLWEToRef,
        S: GLWESecretPreparedToRef<BE>,
        P: GLWEPlaintextToRef,
    {
        let res: &GLWE<&[u8]> = &res.to_ref();
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.glwe_decrypt_tmp_bytes(res));
        let noise_have: f64 = self
            .glwe_noise(res, sk_prepared, pt_want, scratch.borrow())
            .std()
            .log2();
        assert!(noise_have <= max_noise, "{noise_have} {max_noise}");
    }
}
