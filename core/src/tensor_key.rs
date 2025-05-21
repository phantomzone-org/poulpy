use backend::{
    Backend, FFT64, MatZnxDft, MatZnxDftToMut, MatZnxDftToRef, Module, ScalarZnx, ScalarZnxDft, ScalarZnxDftAlloc,
    ScalarZnxDftOps, ScalarZnxDftToRef, Scratch, VecZnxDftOps, VecZnxDftToRef,
};
use sampling::source::Source;

use crate::{
    elem::Infos,
    keys::{SecretKey, SecretKeyFourier},
    keyswitch_key::GLWESwitchingKey,
};

pub struct TensorKey<C, B: Backend> {
    pub(crate) keys: Vec<GLWESwitchingKey<C, B>>,
}

impl TensorKey<Vec<u8>, FFT64> {
    pub fn alloc(module: &Module<FFT64>, basek: usize, k: usize, rows: usize, rank: usize) -> Self {
        let mut keys: Vec<GLWESwitchingKey<Vec<u8>, FFT64>> = Vec::new();
        let pairs: usize = ((rank + 1) * rank) >> 1;
        (0..pairs).for_each(|_| {
            keys.push(GLWESwitchingKey::alloc(module, basek, k, rows, 1, rank));
        });
        Self { keys: keys }
    }
}

impl<T, B: Backend> Infos for TensorKey<T, B> {
    type Inner = MatZnxDft<T, B>;

    fn inner(&self) -> &Self::Inner {
        &self.keys[0].inner()
    }

    fn basek(&self) -> usize {
        self.keys[0].basek()
    }

    fn k(&self) -> usize {
        self.keys[0].k()
    }
}

impl<T, B: Backend> TensorKey<T, B> {
    pub fn rank(&self) -> usize {
        self.keys[0].rank()
    }

    pub fn rank_in(&self) -> usize {
        self.keys[0].rank_in()
    }

    pub fn rank_out(&self) -> usize {
        self.keys[0].rank_out()
    }
}

impl TensorKey<Vec<u8>, FFT64> {
    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, rank: usize, size: usize) -> usize {
        module.bytes_of_scalar_znx_dft(1) + GLWESwitchingKey::encrypt_sk_scratch_space(module, rank, size)
    }
}

impl<DataSelf> TensorKey<DataSelf, FFT64>
where
    MatZnxDft<DataSelf, FFT64>: MatZnxDftToMut<FFT64> + MatZnxDftToRef<FFT64>,
{
    pub fn encrypt_sk<DataSk>(
        &mut self,
        module: &Module<FFT64>,
        sk_dft: &SecretKeyFourier<DataSk, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        ScalarZnxDft<DataSk, FFT64>: VecZnxDftToRef<FFT64> + ScalarZnxDftToRef<FFT64>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), sk_dft.rank());
            assert_eq!(self.n(), module.n());
            assert_eq!(sk_dft.n(), module.n());
        }

        let rank: usize = self.rank();

        (0..rank).for_each(|i| {
            (i..rank).for_each(|j| {
                let (mut sk_ij_dft, scratch1) = scratch.tmp_scalar_znx_dft(module, 1);
                module.svp_apply(&mut sk_ij_dft, 0, &sk_dft.data, i, &sk_dft.data, j);
                let sk_ij: ScalarZnx<&mut [u8]> = module
                    .vec_znx_idft_consume(sk_ij_dft.as_vec_znx_dft())
                    .to_vec_znx_small()
                    .to_scalar_znx();
                let sk_ij: SecretKey<&mut [u8]> = SecretKey {
                    data: sk_ij,
                    dist: sk_dft.dist,
                };

                self.at_mut(i, j).encrypt_sk(
                    module, &sk_ij, sk_dft, source_xa, source_xe, sigma, scratch1,
                );
            });
        })
    }

    // Returns a mutable reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at_mut(&mut self, mut i: usize, mut j: usize) -> &mut GLWESwitchingKey<DataSelf, FFT64> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank();
        &mut self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<DataSelf> TensorKey<DataSelf, FFT64>
where
    MatZnxDft<DataSelf, FFT64>: MatZnxDftToRef<FFT64>,
{
    // Returns a reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at(&self, mut i: usize, mut j: usize) -> &GLWESwitchingKey<DataSelf, FFT64> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank();
        &self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}
