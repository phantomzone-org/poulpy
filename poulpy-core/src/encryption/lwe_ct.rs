use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, ZnAddNormal, ZnFillUniform, ZnNormalizeInplace},
    layouts::{Backend, DataMut, Module, ScratchOwned, Zn, ZnxView, ZnxViewMut},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl},
    source::Source,
};

use crate::{
    encryption::{SIGMA, SIGMA_BOUND},
    layouts::{LWE, LWEInfos, LWEPlaintext, LWEPlaintextToRef, LWESecret, LWESecretToRef, LWEToMut},
};

impl<DataSelf: DataMut> LWE<DataSelf> {
    pub fn encrypt_sk<P, S, M, BE: Backend>(&mut self, module: &M, pt: &P, sk: &S, source_xa: &mut Source, source_xe: &mut Source)
    where
        P: LWEPlaintextToRef,
        S: LWESecretToRef,
        M: LWEEncryptSk<BE>,
        BE: Backend + ScratchOwnedAllocImpl<BE> + ScratchOwnedBorrowImpl<BE>,
    {
        module.lwe_encrypt_sk(self, pt, sk, source_xa, source_xe);
    }
}

pub trait LWEEncryptSk<BE: Backend> {
    fn lwe_encrypt_sk<R, P, S>(&self, res: &mut R, pt: &P, sk: &S, source_xa: &mut Source, source_xe: &mut Source)
    where
        R: LWEToMut,
        P: LWEPlaintextToRef,
        S: LWESecretToRef;
}

impl<BE: Backend> LWEEncryptSk<BE> for Module<BE>
where
    Self: Sized + ZnFillUniform + ZnAddNormal + ZnNormalizeInplace<BE>,
    BE: ScratchOwnedAllocImpl<BE> + ScratchOwnedBorrowImpl<BE>,
{
    fn lwe_encrypt_sk<R, P, S>(&self, res: &mut R, pt: &P, sk: &S, source_xa: &mut Source, source_xe: &mut Source)
    where
        R: LWEToMut,
        P: LWEPlaintextToRef,
        S: LWESecretToRef,
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

        self.zn_fill_uniform((res.n() + 1).into(), base2k, &mut res.data, 0, source_xa);

        let mut tmp_znx: Zn<Vec<u8>> = Zn::alloc(1, 1, res.size());

        let min_size = res.size().min(pt.size());

        (0..min_size).for_each(|i| {
            tmp_znx.at_mut(0, i)[0] = pt.data.at(0, i)[0]
                - res.data.at(0, i)[1..]
                    .iter()
                    .zip(sk.data.at(0, 0))
                    .map(|(x, y)| x * y)
                    .sum::<i64>();
        });

        (min_size..res.size()).for_each(|i| {
            tmp_znx.at_mut(0, i)[0] -= res.data.at(0, i)[1..]
                .iter()
                .zip(sk.data.at(0, 0))
                .map(|(x, y)| x * y)
                .sum::<i64>();
        });

        self.zn_add_normal(
            1,
            base2k,
            &mut res.data,
            0,
            k,
            source_xe,
            SIGMA,
            SIGMA_BOUND,
        );

        self.zn_normalize_inplace(
            1,
            base2k,
            &mut tmp_znx,
            0,
            ScratchOwned::alloc(size_of::<i64>()).borrow(),
        );

        (0..res.size()).for_each(|i| {
            res.data.at_mut(0, i)[0] = tmp_znx.at(0, i)[0];
        });
    }
}
