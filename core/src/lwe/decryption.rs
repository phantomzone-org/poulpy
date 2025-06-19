use backend::{ZnxView, ZnxViewMut, alloc_aligned};

use crate::{Infos, LWECiphertext, LWESecret, SetMetaData, lwe::LWEPlaintext};

impl<DataSelf> LWECiphertext<DataSelf>
where
    DataSelf: AsRef<[u8]>,
{
    pub fn decrypt<DataPt, DataSk>(&self, pt: &mut LWEPlaintext<DataPt>, sk: &LWESecret<DataSk>)
    where
        DataPt: AsRef<[u8]> + AsMut<[u8]>,
        DataSk: AsRef<[u8]>,
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

        let mut tmp_bytes: Vec<u8> = alloc_aligned(size_of::<i64>());
        pt.data.normalize(self.basek(), 0, &mut tmp_bytes);

        pt.set_basek(self.basek());
        pt.set_k(self.k().min(pt.size() * self.basek()));
    }
}
