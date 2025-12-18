use poulpy_hal::{
    api::VecZnxNormalizeInplace,
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxView, ZnxViewMut},
};

use crate::{
    ScratchTakeCore,
    layouts::{LWE, LWEInfos, LWEPlaintext, LWEPlaintextToMut, LWESecret, LWESecretToRef, LWEToMut},
};

impl<DataSelf: DataRef + DataMut> LWE<DataSelf> {
    pub fn decrypt<P, S, M, BE: Backend>(&mut self, module: &M, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        P: LWEPlaintextToMut,
        S: LWESecretToRef,
        M: LWEDecrypt<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.lwe_decrypt(self, pt, sk, scratch);
    }
}

pub trait LWEDecrypt<BE: Backend> {
    fn lwe_decrypt<R, P, S>(&self, res: &mut R, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        R: LWEToMut,
        P: LWEPlaintextToMut,
        S: LWESecretToRef,
        Scratch<BE>: ScratchTakeCore<BE>;
}

impl<BE: Backend> LWEDecrypt<BE> for Module<BE>
where
    Self: Sized + VecZnxNormalizeInplace<BE>,
{
    fn lwe_decrypt<R, P, S>(&self, res: &mut R, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        R: LWEToMut,
        P: LWEPlaintextToMut,
        S: LWESecretToRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut LWE<&mut [u8]> = &mut res.to_mut();
        let pt: &mut LWEPlaintext<&mut [u8]> = &mut pt.to_mut();
        let sk: LWESecret<&[u8]> = sk.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), sk.n());
        }

        (0..pt.size().min(res.size())).for_each(|i| {
            pt.data.at_mut(0, i)[0] = res.data.at(0, i)[0]
                + res.data.at(0, i)[1..]
                    .iter()
                    .zip(sk.data.at(0, 0))
                    .map(|(x, y)| x * y)
                    .sum::<i64>();
        });
        self.vec_znx_normalize_inplace(res.base2k().into(), &mut pt.data, 0, scratch);
        pt.base2k = res.base2k();
        pt.k = crate::layouts::TorusPrecision(res.k().0.min(pt.size() as u32 * res.base2k().0));
    }
}
