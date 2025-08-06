use backend::hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalizeInplace, ZnxView, ZnxViewMut},
    layouts::{Backend, DataMut, DataRef, Module, ScratchOwned},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl},
};

use crate::{Infos, LWECiphertext, LWESecret, SetMetaData, lwe::LWEPlaintext};

impl<DataSelf> LWECiphertext<DataSelf>
where
    DataSelf: DataRef,
{
    pub fn decrypt<DataPt, DataSk, B: Backend>(&self, module: &Module<B>, pt: &mut LWEPlaintext<DataPt>, sk: &LWESecret<DataSk>)
    where
        DataPt: DataMut,
        DataSk: DataRef,
        Module<B>: VecZnxNormalizeInplace<B>,
        B: ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), sk.n());
        }

        (0..pt.size().min(self.size())).for_each(|i| {
            pt.data.at_mut(0, i)[0] = self.data.at(0, i)[0]
                + self.data.at(0, i)[1..]
                    .iter()
                    .zip(sk.data.at(0, 0))
                    .map(|(x, y)| x * y)
                    .sum::<i64>();
        });
        module.vec_znx_normalize_inplace(
            self.basek(),
            &mut pt.data,
            0,
            ScratchOwned::alloc(size_of::<i64>()).borrow(),
        );
        pt.set_basek(self.basek());
        pt.set_k(self.k().min(pt.size() * self.basek()));
    }
}
