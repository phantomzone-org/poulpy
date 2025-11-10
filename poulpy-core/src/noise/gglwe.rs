use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, ScratchTakeBasic, VecZnxFillUniform, VecZnxSubScalarInplace},
    layouts::{Backend, DataRef, Module, ScalarZnxToRef, Scratch, ScratchOwned, ZnxZero},
};

use crate::decryption::GLWEDecrypt;
use crate::layouts::{GGLWE, GGLWEInfos, GGLWEToRef, GLWEPlaintext, LWEInfos, prepared::GLWESecretPreparedToRef};

impl<D: DataRef> GGLWE<D> {
    pub fn assert_noise<M, S, P, BE: Backend>(&self, module: &M, sk_prepared: &S, pt_want: &P, max_noise: f64)
    where
        S: GLWESecretPreparedToRef<BE>,
        P: ScalarZnxToRef,
        M: GGLWENoise<BE>,
        Scratch<BE>: ScratchTakeBasic,
    {
        module.gglwe_assert_noise(self, sk_prepared, pt_want, max_noise);
    }
}

pub trait GGLWENoise<BE: Backend> {
    fn gglwe_assert_noise<R, S, P>(&self, res: &R, sk_prepared: &S, pt_want: &P, max_noise: f64)
    where
        R: GGLWEToRef,
        S: GLWESecretPreparedToRef<BE>,
        P: ScalarZnxToRef,
        Scratch<BE>: ScratchTakeBasic;
}

impl<BE: Backend> GGLWENoise<BE> for Module<BE>
where
    Module<BE>: GLWEDecrypt<BE> + VecZnxFillUniform + VecZnxSubScalarInplace,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeBasic,
{
    fn gglwe_assert_noise<R, S, P>(&self, res: &R, sk_prepared: &S, pt_want: &P, max_noise: f64)
    where
        R: GGLWEToRef,
        S: GLWESecretPreparedToRef<BE>,
        P: ScalarZnxToRef,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeBasic,
    {
        let res: &GGLWE<&[u8]> = &res.to_ref();

        let dsize: usize = res.dsize().into();
        let base2k: usize = res.base2k().into();

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.glwe_decrypt_tmp_bytes(res));
        let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(res);

        (0..res.rank_in().into()).for_each(|col_i| {
            (0..res.dnum().into()).for_each(|row_i| {
                self.glwe_decrypt(
                    &res.at(row_i, col_i),
                    &mut pt,
                    sk_prepared,
                    scratch.borrow(),
                );

                self.vec_znx_sub_scalar_inplace(&mut pt.data, 0, (dsize - 1) + row_i * dsize, pt_want, col_i);

                let noise_have: f64 = pt.data.stats(base2k, 0).std().log2();

                println!("noise_have: {noise_have}");

                assert!(
                    noise_have <= max_noise,
                    "noise_have: {noise_have} > max_noise: {max_noise}"
                );

                pt.data.zero();
            });
        });
    }
}
