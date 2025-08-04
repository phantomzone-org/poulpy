use backend::hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddNormal, VecZnxFillUniform, VecZnxNormalizeInplace, ZnxView, ZnxViewMut,
    },
    layouts::{Backend, Module, ScratchOwned, VecZnx},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl},
};
use sampling::source::Source;

use crate::{Infos, LWECiphertext, LWESecret, SIX_SIGMA, lwe::LWEPlaintext};

impl<DataSelf> LWECiphertext<DataSelf>
where
    DataSelf: AsMut<[u8]> + AsRef<[u8]>,
{
    pub fn encrypt_sk<DataPt, DataSk, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: &LWEPlaintext<DataPt>,
        sk: &LWESecret<DataSk>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
    ) where
        DataPt: AsRef<[u8]>,
        DataSk: AsRef<[u8]>,
        Module<B>: VecZnxFillUniform + VecZnxAddNormal + VecZnxNormalizeInplace<B>,
        B: ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), sk.n())
        }

        let basek: usize = self.basek();
        let k: usize = self.k();

        module.vec_znx_fill_uniform(basek, &mut self.data, 0, k, source_xa);

        let mut tmp_znx: VecZnx<Vec<u8>> = VecZnx::<Vec<u8>>::new::<i64>(1, 1, self.size());

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

        module.vec_znx_add_normal(
            basek,
            &mut self.data,
            0,
            k,
            source_xe,
            sigma,
            sigma * SIX_SIGMA,
        );

        module.vec_znx_normalize_inplace(
            basek,
            &mut tmp_znx,
            0,
            ScratchOwned::alloc(size_of::<i64>()).borrow(),
        );

        (0..self.size()).for_each(|i| {
            self.data.at_mut(0, i)[0] = tmp_znx.at(0, i)[0];
        });
    }
}
