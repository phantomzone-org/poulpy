use poulpy_hal::{
    api::{
        ModuleN, ScratchTakeBasic, SvpApplyDftToDftInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigBytesOf,
        VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyConsume, VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, DataMut, DataViewMut, Module, Scratch},
};

use crate::layouts::{
    GLWE, GLWEInfos, GLWEPlaintext, GLWEPlaintextToMut, GLWEToRef, LWEInfos,
    prepared::{GLWESecretPrepared, GLWESecretPreparedToRef},
};

impl GLWE<Vec<u8>> {
    pub fn decrypt_tmp_bytes<A, M, BE: Backend>(module: &M, a_infos: &A) -> usize
    where
        A: GLWEInfos,
        M: GLWEDecrypt<BE>,
    {
        module.glwe_decrypt_tmp_bytes(a_infos)
    }
}

impl<DataSelf: DataMut> GLWE<DataSelf> {
    pub fn decrypt<P, S, M, BE: Backend>(&mut self, module: &M, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        P: GLWEPlaintextToMut,
        S: GLWESecretPreparedToRef<BE>,
        M: GLWEDecrypt<BE>,
        Scratch<BE>: ScratchTakeBasic,
    {
        module.glwe_decrypt(self, pt, sk, scratch);
    }
}

pub trait GLWEDecrypt<BE: Backend>
where
    Self: Sized
        + ModuleN
        + VecZnxDftBytesOf
        + VecZnxNormalizeTmpBytes
        + VecZnxBigBytesOf
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftInplace<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigAddInplace<BE>
        + VecZnxBigAddSmallInplace<BE>
        + VecZnxBigNormalize<BE>,
{
    fn glwe_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let size: usize = infos.size();
        (self.vec_znx_normalize_tmp_bytes() | self.bytes_of_vec_znx_dft(1, size)) + self.bytes_of_vec_znx_dft(1, size)
    }

    fn glwe_decrypt<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        R: GLWEToRef,
        P: GLWEPlaintextToMut,
        S: GLWESecretPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeBasic,
    {
        let res: &GLWE<&[u8]> = &res.to_ref();
        let pt: &mut GLWEPlaintext<&mut [u8]> = &mut pt.to_ref();
        let sk: &GLWESecretPrepared<&[u8], BE> = &sk.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.rank(), sk.rank());
            assert_eq!(res.n(), sk.n());
            assert_eq!(pt.n(), sk.n());
        }

        let cols: usize = (res.rank() + 1).into();

        let (mut c0_big, scratch_1) = scratch.take_vec_znx_big(self, 1, res.size()); // TODO optimize size when pt << ct
        c0_big.data_mut().fill(0);

        {
            (1..cols).for_each(|i| {
                // ci_dft = DFT(a[i]) * DFT(s[i])
                let (mut ci_dft, _) = scratch_1.take_vec_znx_dft(self, 1, res.size()); // TODO optimize size when pt << ct
                self.vec_znx_dft_apply(1, 0, &mut ci_dft, 0, &res.data, i);
                self.svp_apply_dft_to_dft_inplace(&mut ci_dft, 0, &sk.data, i - 1);
                let ci_big = self.vec_znx_idft_apply_consume(ci_dft);

                // c0_big += a[i] * s[i]
                self.vec_znx_big_add_inplace(&mut c0_big, 0, &ci_big, 0);
            });
        }

        // c0_big = (a * s) + (-a * s + m + e) = BIG(m + e)
        self.vec_znx_big_add_small_inplace(&mut c0_big, 0, &res.data, 0);

        // pt = norm(BIG(m + e))
        self.vec_znx_big_normalize(
            res.base2k().into(),
            &mut pt.data,
            0,
            res.base2k().into(),
            &c0_big,
            0,
            scratch_1,
        );

        pt.base2k = res.base2k();
        pt.k = pt.k().min(res.k());
    }
}

impl<BE: Backend> GLWEDecrypt<BE> for Module<BE> where
    Self: ModuleN
        + VecZnxDftBytesOf
        + VecZnxNormalizeTmpBytes
        + VecZnxBigBytesOf
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftInplace<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigAddInplace<BE>
        + VecZnxBigAddSmallInplace<BE>
        + VecZnxBigNormalize<BE>
{
}
