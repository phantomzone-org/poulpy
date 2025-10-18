use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, ScratchTakeBasic, SvpApplyDftToDftInplace, VecZnxAddScalarInplace,
        VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigAlloc, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyConsume,
        VecZnxIdftApplyTmpA, VecZnxNormalizeTmpBytes, VecZnxSubInplace,
    },
    layouts::{Backend, DataRef, Module, ScalarZnx, ScalarZnxToRef, Scratch, ScratchOwned, VecZnxBig, VecZnxDft, ZnxZero},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl},
};

use crate::decryption::GLWEDecryption;
use crate::layouts::prepared::GLWESecretPreparedToRef;
use crate::layouts::{GGSW, GGSWInfos, GGSWToRef, GLWEInfos, GLWEPlaintext, LWEInfos, prepared::GLWESecretPrepared};

impl<D: DataRef> GGSW<D> {
    pub fn assert_noise<M, BE, DataSk, DataScalar, F>(
        &self,
        module: &M,
        sk_prepared: &GLWESecretPrepared<DataSk, BE>,
        pt_want: &ScalarZnx<DataScalar>,
        max_noise: F,
    ) where
        DataSk: DataRef,
        DataScalar: DataRef,
        M: GGSWNoise<BE>,
        BE: Backend + ScratchOwnedAllocImpl<BE> + ScratchOwnedBorrowImpl<BE> + ScratchOwnedBorrow<BE>,
        F: Fn(usize) -> f64,
    {
        module.ggsw_assert_noise(self, sk_prepared, pt_want, max_noise);
    }

    pub fn print_noise<M, BE, DataSk, DataScalar>(
        &self,
        module: &M,
        sk_prepared: &GLWESecretPrepared<DataSk, BE>,
        pt_want: &ScalarZnx<DataScalar>,
    ) where
        DataSk: DataRef,
        DataScalar: DataRef,
        M: GGSWNoise<BE>,
        BE: Backend + ScratchOwnedAllocImpl<BE> + ScratchOwnedBorrowImpl<BE> + ScratchOwnedBorrow<BE>,
    {
        module.ggsw_print_noise(self, sk_prepared, pt_want);
    }
}

pub trait GGSWNoise<BE: Backend> {
    fn ggsw_assert_noise<R, S, P, F>(&self, res: &R, sk_prepared: &S, pt_want: &P, max_noise: F)
    where
        R: GGSWToRef,
        S: GLWESecretPreparedToRef<BE>,
        P: ScalarZnxToRef,
        BE: ScratchOwnedAllocImpl<BE> + ScratchOwnedBorrowImpl<BE> + ScratchOwnedBorrow<BE>,
        F: Fn(usize) -> f64;

    fn ggsw_print_noise<R, S, P>(&self, res: &R, sk_prepared: &S, pt_want: &P)
    where
        R: GGSWToRef,
        S: GLWESecretPreparedToRef<BE>,
        P: ScalarZnxToRef,
        BE: ScratchOwnedAllocImpl<BE> + ScratchOwnedBorrowImpl<BE> + ScratchOwnedBorrow<BE>;
}

impl<BE: Backend> GGSWNoise<BE> for Module<BE>
where
    Module<BE>: VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftInplace<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigAddInplace<BE>
        + VecZnxBigAddSmallInplace<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxBigAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxAddScalarInplace
        + VecZnxSubInplace
        + GLWEDecryption<BE>,
    Scratch<BE>: ScratchTakeBasic,
{
    fn ggsw_assert_noise<R, S, P, F>(&self, res: &R, sk_prepared: &S, pt_want: &P, max_noise: F)
    where
        R: GGSWToRef,
        S: GLWESecretPreparedToRef<BE>,
        P: ScalarZnxToRef,
        BE: ScratchOwnedAllocImpl<BE> + ScratchOwnedBorrowImpl<BE> + ScratchOwnedBorrow<BE>,
        F: Fn(usize) -> f64,
    {
        let res: &GGSW<&[u8]> = &res.to_ref();
        let sk_prepared: &GLWESecretPrepared<&[u8], BE> = &sk_prepared.to_ref();

        let base2k: usize = res.base2k().into();
        let dsize: usize = res.dsize().into();

        let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(self, res);
        let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(self, res);
        let mut pt_dft: VecZnxDft<Vec<u8>, BE> = self.vec_znx_dft_alloc(1, res.size());
        let mut pt_big: VecZnxBig<Vec<u8>, BE> = self.vec_znx_big_alloc(1, res.size());

        let mut scratch: ScratchOwned<BE> =
            ScratchOwned::alloc(self.glwe_decrypt_tmp_bytes(res) | self.vec_znx_normalize_tmp_bytes());

        (0..(res.rank() + 1).into()).for_each(|col_j| {
            (0..res.dnum().into()).for_each(|row_i| {
                self.vec_znx_add_scalar_inplace(&mut pt.data, 0, (dsize - 1) + row_i * dsize, pt_want, 0);

                // mul with sk[col_j-1]
                if col_j > 0 {
                    self.vec_znx_dft_apply(1, 0, &mut pt_dft, 0, &pt.data, 0);
                    self.svp_apply_dft_to_dft_inplace(&mut pt_dft, 0, &sk_prepared.data, col_j - 1);
                    self.vec_znx_idft_apply_tmpa(&mut pt_big, 0, &mut pt_dft, 0);
                    self.vec_znx_big_normalize(
                        base2k,
                        &mut pt.data,
                        0,
                        base2k,
                        &pt_big,
                        0,
                        scratch.borrow(),
                    );
                }

                self.glwe_decrypt(
                    &res.at(row_i, col_j),
                    &mut pt_have,
                    sk_prepared,
                    scratch.borrow(),
                );

                self.vec_znx_sub_inplace(&mut pt_have.data, 0, &pt.data, 0);

                let std_pt: f64 = pt_have.data.std(base2k, 0).log2();
                let noise: f64 = max_noise(col_j);
                assert!(std_pt <= noise, "{std_pt} > {noise}");

                pt.data.zero();
            });
        });
    }

    fn ggsw_print_noise<R, S, P>(&self, res: &R, sk_prepared: &S, pt_want: &P)
    where
        R: GGSWToRef,
        S: GLWESecretPreparedToRef<BE>,
        P: ScalarZnxToRef,
        BE: ScratchOwnedAllocImpl<BE> + ScratchOwnedBorrowImpl<BE> + ScratchOwnedBorrow<BE>,
    {
        let res: &GGSW<&[u8]> = &res.to_ref();
        let sk_prepared: &GLWESecretPrepared<&[u8], BE> = &sk_prepared.to_ref();

        let base2k: usize = res.base2k().into();
        let dsize: usize = res.dsize().into();

        let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(self, res);
        let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(self, res);
        let mut pt_dft: VecZnxDft<Vec<u8>, BE> = self.vec_znx_dft_alloc(1, res.size());
        let mut pt_big: VecZnxBig<Vec<u8>, BE> = self.vec_znx_big_alloc(1, res.size());

        let mut scratch: ScratchOwned<BE> =
            ScratchOwned::alloc(self.glwe_decrypt_tmp_bytes(res) | self.vec_znx_normalize_tmp_bytes());

        (0..(res.rank() + 1).into()).for_each(|col_j| {
            (0..res.dnum().into()).for_each(|row_i| {
                self.vec_znx_add_scalar_inplace(&mut pt.data, 0, (dsize - 1) + row_i * dsize, pt_want, 0);

                // mul with sk[col_j-1]
                if col_j > 0 {
                    self.vec_znx_dft_apply(1, 0, &mut pt_dft, 0, &pt.data, 0);
                    self.svp_apply_dft_to_dft_inplace(&mut pt_dft, 0, &sk_prepared.data, col_j - 1);
                    self.vec_znx_idft_apply_tmpa(&mut pt_big, 0, &mut pt_dft, 0);
                    self.vec_znx_big_normalize(
                        base2k,
                        &mut pt.data,
                        0,
                        base2k,
                        &pt_big,
                        0,
                        scratch.borrow(),
                    );
                }

                self.glwe_decrypt(
                    &res.at(row_i, col_j),
                    &mut pt_have,
                    sk_prepared,
                    scratch.borrow(),
                );
                self.vec_znx_sub_inplace(&mut pt_have.data, 0, &pt.data, 0);

                let std_pt: f64 = pt_have.data.std(base2k, 0).log2();
                println!("col: {col_j} row: {row_i}: {std_pt}");
                pt.data.zero();
            });
        });
    }
}
