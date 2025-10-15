use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, ZnNormalizeInplace},
    layouts::{Backend, DataMut, DataRef, Module, ScratchOwned, ZnxView, ZnxViewMut},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl},
};

use crate::layouts::{LWE, LWEInfos, LWEPlaintext, LWESecret, LWEToMut, LWEPlaintextToMut, LWESecretToRef};

impl<DataSelf: DataRef + DataMut> LWE<DataSelf>
{
    pub fn decrypt<P, S, M, B>(&mut self, module: &M, pt: &mut P, sk: S)
    where
        P: LWEPlaintextToMut,
        S: LWESecretToRef,
        M: LWEDecrypt<B>,
        B: Backend + ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
    {
        module.lwe_decrypt(self, pt, sk);
    }
}

pub trait LWEDecrypt<BE: Backend>
where
    Self: Sized + ZnNormalizeInplace<BE>
{
    fn lwe_decrypt<R, P, S>(&self, res: &mut R, pt: &mut P, sk: S)
    where
            R: LWEToMut,
            P: LWEPlaintextToMut,
            S: LWESecretToRef,
            BE: Backend + ScratchOwnedAllocImpl<BE> + ScratchOwnedBorrowImpl<BE>,
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
        self.zn_normalize_inplace(
            1,
            res.base2k().into(),
            &mut pt.data,
            0,
            ScratchOwned::alloc(size_of::<i64>()).borrow(),
        );
        pt.base2k = res.base2k();
        pt.k = crate::layouts::TorusPrecision(res.k().0.min(pt.size() as u32 * res.base2k().0));
    }
}

impl<BE: Backend> LWEDecrypt<BE> for Module<BE> where 
    Self: Sized + ZnNormalizeInplace<BE>
{
    
}