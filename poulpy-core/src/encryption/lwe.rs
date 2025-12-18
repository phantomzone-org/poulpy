use poulpy_hal::{
    api::{VecZnxAddNormal, VecZnxFillUniform, VecZnxNormalizeInplace},
    layouts::{Backend, DataMut, Module, Scratch, VecZnx, ZnxView, ZnxViewMut},
    source::Source,
};

use crate::{
    ScratchTakeCore,
    encryption::{SIGMA, SIGMA_BOUND},
    layouts::{LWE, LWEInfos, LWEPlaintext, LWEPlaintextToRef, LWESecret, LWESecretToRef, LWEToMut},
};

impl<DataSelf: DataMut> LWE<DataSelf> {
    pub fn encrypt_sk<P, S, M, BE: Backend>(
        &mut self,
        module: &M,
        pt: &P,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        P: LWEPlaintextToRef,
        S: LWESecretToRef,
        M: LWEEncryptSk<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.lwe_encrypt_sk(self, pt, sk, source_xa, source_xe, scratch);
    }
}

pub trait LWEEncryptSk<BE: Backend> {
    fn lwe_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: LWEToMut,
        P: LWEPlaintextToRef,
        S: LWESecretToRef,
        Scratch<BE>: ScratchTakeCore<BE>;
}

impl<BE: Backend> LWEEncryptSk<BE> for Module<BE>
where
    Self: Sized + VecZnxFillUniform + VecZnxAddNormal + VecZnxNormalizeInplace<BE>,
{
    fn lwe_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: LWEToMut,
        P: LWEPlaintextToRef,
        S: LWESecretToRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut LWE<&mut [u8]> = &mut res.to_mut();
        let pt: &LWEPlaintext<&[u8]> = &pt.to_ref();
        let sk: &LWESecret<&[u8]> = &sk.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), sk.n())
        }

        let base2k: usize = res.base2k().into();
        let k: usize = res.k().into();

        self.vec_znx_fill_uniform(base2k, &mut res.data, 0, source_xa);

        let mut tmp_znx: VecZnx<Vec<u8>> = VecZnx::alloc(1, 1, res.limbs());

        let min_size: usize = res.limbs().min(pt.limbs());

        (0..min_size).for_each(|i| {
            tmp_znx.at_mut(0, i)[0] = pt.data.at(0, i)[0]
                - res.data.at(0, i)[1..]
                    .iter()
                    .zip(sk.data.at(0, 0))
                    .map(|(x, y)| x * y)
                    .sum::<i64>();
        });

        (min_size..res.limbs()).for_each(|i| {
            tmp_znx.at_mut(0, i)[0] -= res.data.at(0, i)[1..]
                .iter()
                .zip(sk.data.at(0, 0))
                .map(|(x, y)| x * y)
                .sum::<i64>();
        });

        self.vec_znx_add_normal(base2k, &mut tmp_znx, 0, k, source_xe, SIGMA, SIGMA_BOUND);

        self.vec_znx_normalize_inplace(base2k, &mut tmp_znx, 0, scratch);

        (0..res.limbs()).for_each(|i| {
            res.data.at_mut(0, i)[0] = tmp_znx.at(0, i)[0];
        });
    }
}
