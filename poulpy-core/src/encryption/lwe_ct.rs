use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, ZnAddNormal, ZnFillUniform, ZnNormalizeInplace},
    layouts::{Backend, DataMut, DataRef, Module, ScratchOwned, Zn, ZnxView, ZnxViewMut},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl},
    source::Source,
};

use crate::{
    encryption::{SIGMA, SIGMA_BOUND},
    layouts::{LWECiphertext, LWEInfos, LWEPlaintext, LWESecret},
};

impl<DataSelf: DataMut> LWECiphertext<DataSelf> {
    pub fn encrypt_sk<DataPt, DataSk, B>(
        &mut self,
        module: &Module<B>,
        pt: &LWEPlaintext<DataPt>,
        sk: &LWESecret<DataSk>,
        source_xa: &mut Source,
        source_xe: &mut Source,
    ) where
        DataPt: DataRef,
        DataSk: DataRef,
        Module<B>: ZnFillUniform + ZnAddNormal + ZnNormalizeInplace<B>,
        B: Backend + ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), sk.n())
        }

        let base2k: usize = self.base2k().into();
        let k: usize = self.k().into();

        module.zn_fill_uniform((self.n() + 1).into(), base2k, &mut self.data, 0, source_xa);

        let mut tmp_znx: Zn<Vec<u8>> = Zn::alloc(1, 1, self.size());

        let min_size = self.size().min(pt.size());

        (0..min_size).for_each(|i| {
            tmp_znx.at_mut(0, i)[0] = pt.data.at(0, i)[0]
                - self.data.at(0, i)[1..]
                    .iter()
                    .zip(sk.data.at(0, 0))
                    .map(|(x, y)| x * y)
                    .sum::<i64>();
        });

        (min_size..self.size()).for_each(|i| {
            tmp_znx.at_mut(0, i)[0] -= self.data.at(0, i)[1..]
                .iter()
                .zip(sk.data.at(0, 0))
                .map(|(x, y)| x * y)
                .sum::<i64>();
        });

        module.zn_add_normal(
            1,
            base2k,
            &mut self.data,
            0,
            k,
            source_xe,
            SIGMA,
            SIGMA_BOUND,
        );

        module.zn_normalize_inplace(
            1,
            base2k,
            &mut tmp_znx,
            0,
            ScratchOwned::alloc(size_of::<i64>()).borrow(),
        );

        (0..self.size()).for_each(|i| {
            self.data.at_mut(0, i)[0] = tmp_znx.at(0, i)[0];
        });
    }
}
