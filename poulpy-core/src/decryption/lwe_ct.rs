use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, ZnNormalizeInplace},
    layouts::{Backend, DataMut, DataRef, Module, ScratchOwned, ZnxView, ZnxViewMut},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl},
};

use crate::layouts::{LWE, LWEInfos, LWEPlaintext, LWESecret};

impl<DataSelf> LWE<DataSelf>
where
    DataSelf: DataRef,
{
    pub fn decrypt<DataPt, DataSk, B>(&self, module: &Module<B>, pt: &mut LWEPlaintext<DataPt>, sk: &LWESecret<DataSk>)
    where
        DataPt: DataMut,
        DataSk: DataRef,
        Module<B>: ZnNormalizeInplace<B>,
        B: Backend + ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
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
        module.zn_normalize_inplace(
            1,
            self.base2k().into(),
            &mut pt.data,
            0,
            ScratchOwned::alloc(size_of::<i64>()).borrow(),
        );
        pt.base2k = self.base2k();
        pt.k = crate::layouts::TorusPrecision(self.k().0.min(pt.size() as u32 * self.base2k().0));
    }
}
