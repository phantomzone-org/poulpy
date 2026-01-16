use poulpy_hal::{
    api::{VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxView, ZnxViewMut},
};

use crate::{
    ScratchTakeCore,
    layouts::{LWE, LWEInfos, LWEPlaintext, LWEPlaintextToMut, LWESecret, LWESecretToRef, LWEToRef, SetLWEInfos, TorusPrecision},
};

impl<DataSelf: DataRef + DataMut> LWE<DataSelf> {
    pub fn decrypt<P, S, M, BE: Backend>(&self, module: &M, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        P: LWEPlaintextToMut + SetLWEInfos + LWEInfos,
        S: LWESecretToRef,
        M: LWEDecrypt<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.lwe_decrypt(self, pt, sk, scratch);
    }
}

pub trait LWEDecrypt<BE: Backend> {
    fn lwe_decrypt<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        R: LWEToRef,
        P: LWEPlaintextToMut + SetLWEInfos + LWEInfos,
        S: LWESecretToRef,
        Scratch<BE>: ScratchTakeCore<BE>;
    fn lwe_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos;
}

impl<BE: Backend> LWEDecrypt<BE> for Module<BE>
where
    Self: Sized + VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes,
{
    fn lwe_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos,
    {
        self.vec_znx_normalize_tmp_bytes() + LWEPlaintext::bytes_of(infos.size())
    }

    fn lwe_decrypt<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        R: LWEToRef,
        P: LWEPlaintextToMut + SetLWEInfos + LWEInfos,
        S: LWESecretToRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &LWE<&[u8]> = &res.to_ref();
        let sk: LWESecret<&[u8]> = sk.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), sk.n());
        }

        let (mut tmp, scratch_1) = scratch.take_lwe_plaintext(res);
        for i in 0..res.size() {
            tmp.data.at_mut(0, i)[0] = res.data.at(0, i)[0]
                + res.data.at(0, i)[1..]
                    .iter()
                    .zip(sk.data.at(0, 0))
                    .map(|(x, y)| x * y)
                    .sum::<i64>();
        }

        let pt_base2k = pt.base2k().into();
        let res_base2k = res.base2k().into();
        self.vec_znx_normalize(&mut pt.to_mut().data, pt_base2k, 0, 0, tmp.data(), res_base2k, 0, scratch_1);

        pt.set_k(TorusPrecision(res.k().0.min(pt.size() as u32 * res.base2k().0)));
    }
}
