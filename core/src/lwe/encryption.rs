use backend::{alloc_aligned, AddNormal, FillUniform, VecZnx, ZnxView, ZnxViewMut};
use sampling::source::Source;

use crate::{lwe::LWEPlaintext, Infos, LWECiphertext, LWESecret, SIX_SIGMA};



impl<DataSelf> LWECiphertext<DataSelf> where DataSelf: AsMut<[u8]> + AsRef<[u8]>{
    pub fn encrypt_sk<DataPt, DataSk>(&mut self, pt: &LWEPlaintext<DataPt>, sk: &LWESecret<DataSk>, source_xa: &mut Source, source_xe: &mut Source, sigma: f64) where DataPt: AsRef<[u8]>, DataSk: AsRef<[u8]>{

        #[cfg(debug_assertions)]{
            assert_eq!(self.n(), sk.n())
        }

        let basek: usize = self.basek();

        self.data.fill_uniform(basek, 0, self.size(), source_xa);
        let mut tmp_znx: VecZnx<Vec<u8>> = VecZnx::<Vec<u8>>::new::<i64>(1, 1, self.size());

        (0..self.size()).for_each(|i|{
            tmp_znx.at_mut(0, i)[0] = pt.data.at(0, i)[0] - self.data.at(0, i)[1..].iter().zip(sk.data.at(0, 0)).map(|(x, y)| x * y).sum::<i64>();
        });

        tmp_znx.add_normal(basek, 0, self.k(), source_xe, sigma, sigma*SIX_SIGMA);

        let mut tmp_bytes: Vec<u8> = alloc_aligned(size_of::<i64>());

        tmp_znx.normalize(basek, 0, &mut tmp_bytes);

        (0..self.size()).for_each(|i|{
            self.data.at_mut(0, i)[0] = tmp_znx.at(0, i)[0];
        });

    }
}
