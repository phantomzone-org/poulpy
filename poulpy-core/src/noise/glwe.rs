use poulpy_hal::layouts::{Backend, DataRef, Module, Scratch, Stats};

use crate::{
    GLWENormalize, GLWESub, ScratchTakeCore,
    decryption::GLWEDecrypt,
    layouts::{GLWE, GLWEInfos, GLWEPlaintext, GLWEToRef, LWEInfos, prepared::GLWESecretPreparedToRef},
};

impl<D: DataRef> GLWE<D> {
    pub fn noise<M, P, S, BE: Backend>(&self, module: &M, pt_want: &P, sk_prepared: &S, scratch: &mut Scratch<BE>) -> Stats
    where
        M: GLWENoise<BE>,
        P: GLWEToRef,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
    {
        module.glwe_noise(self, pt_want, sk_prepared, scratch)
    }
}

pub trait GLWENoise<BE: Backend> {
    fn glwe_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_noise<R, P, S>(&self, res: &R, pt_want: &P, sk_prepared: &S, scratch: &mut Scratch<BE>) -> Stats
    where
        R: GLWEToRef + GLWEInfos,
        P: GLWEToRef,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos;
}

impl<BE: Backend> GLWENoise<BE> for Module<BE>
where
    Module<BE>: GLWEDecrypt<BE> + GLWESub + GLWENormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        GLWEPlaintext::bytes_of_from_infos(infos) + self.glwe_normalize_tmp_bytes().max(self.glwe_decrypt_tmp_bytes(infos))
    }

    fn glwe_noise<R, P, S>(&self, res: &R, pt_want: &P, sk_prepared: &S, scratch: &mut Scratch<BE>) -> Stats
    where
        R: GLWEToRef + GLWEInfos,
        P: GLWEToRef,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
    {
        let (mut pt_have, scratch_1) = scratch.take_glwe_plaintext(res);
        self.glwe_decrypt(res, &mut pt_have, sk_prepared, scratch_1);
        self.glwe_sub_inplace(&mut pt_have, pt_want);
        self.glwe_normalize_inplace(&mut pt_have, scratch_1);
        pt_have.data.stats(pt_have.base2k().into(), 0)
    }
}
