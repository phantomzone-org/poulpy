use base2k::{
    Backend, FFT64, MatZnxDft, MatZnxDftOps, MatZnxDftToMut, MatZnxDftToRef, Module, ScalarZnx, ScalarZnxDftOps, ScalarZnxOps,
    ScalarZnxToRef, Scratch, VecZnx, VecZnxBigAlloc, VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxDftToMut, VecZnxDftToRef,
    VecZnxOps, ZnxZero,
};
use sampling::source::Source;

use crate::{
    elem::{GetRow, Infos, SetRow},
    gglwe_ciphertext::GGLWECiphertext,
    ggsw_ciphertext::GGSWCiphertext,
    glwe_ciphertext::GLWECiphertext,
    glwe_ciphertext_fourier::GLWECiphertextFourier,
    keys::{SecretKey, SecretKeyFourier},
    keyswitch_key::GLWESwitchingKey,
};

pub struct AutomorphismKey<Data, B: Backend> {
    pub(crate) key: GLWESwitchingKey<Data, B>,
    pub(crate) p: i64,
}

impl AutomorphismKey<Vec<u8>, FFT64> {
    pub fn new(module: &Module<FFT64>, basek: usize, k: usize, rows: usize, rank: usize) -> Self {
        AutomorphismKey {
            key: GLWESwitchingKey::new(module, basek, k, rows, rank, rank),
            p: 0,
        }
    }
}

impl<T, B: Backend> Infos for AutomorphismKey<T, B> {
    type Inner = MatZnxDft<T, B>;

    fn inner(&self) -> &Self::Inner {
        &self.key.inner()
    }

    fn basek(&self) -> usize {
        self.key.basek()
    }

    fn k(&self) -> usize {
        self.key.k()
    }
}

impl<T, B: Backend> AutomorphismKey<T, B> {
    pub fn p(&self) -> i64 {
        self.p
    }

    pub fn rank(&self) -> usize {
        self.key.rank()
    }

    pub fn rank_in(&self) -> usize {
        self.key.rank_in()
    }

    pub fn rank_out(&self) -> usize {
        self.key.rank_out()
    }
}

impl<DataSelf, B: Backend> MatZnxDftToMut<B> for AutomorphismKey<DataSelf, B>
where
    MatZnxDft<DataSelf, B>: MatZnxDftToMut<B>,
{
    fn to_mut(&mut self) -> MatZnxDft<&mut [u8], B> {
        self.key.to_mut()
    }
}

impl<DataSelf, B: Backend> MatZnxDftToRef<B> for AutomorphismKey<DataSelf, B>
where
    MatZnxDft<DataSelf, B>: MatZnxDftToRef<B>,
{
    fn to_ref(&self) -> MatZnxDft<&[u8], B> {
        self.key.to_ref()
    }
}

impl<C> GetRow<FFT64> for AutomorphismKey<C, FFT64>
where
    MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64>,
{
    fn get_row<R>(&self, module: &Module<FFT64>, row_i: usize, col_j: usize, res: &mut GLWECiphertextFourier<R, FFT64>)
    where
        VecZnxDft<R, FFT64>: VecZnxDftToMut<FFT64>,
    {
        module.vmp_extract_row(res, self, row_i, col_j);
    }
}

impl<C> SetRow<FFT64> for AutomorphismKey<C, FFT64>
where
    MatZnxDft<C, FFT64>: MatZnxDftToMut<FFT64>,
{
    fn set_row<R>(&mut self, module: &Module<FFT64>, row_i: usize, col_j: usize, a: &GLWECiphertextFourier<R, FFT64>)
    where
        VecZnxDft<R, FFT64>: VecZnxDftToRef<FFT64>,
    {
        module.vmp_prepare_row(self, row_i, col_j, a);
    }
}

impl AutomorphismKey<Vec<u8>, FFT64> {
    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, rank: usize, size: usize) -> usize {
        GGLWECiphertext::encrypt_sk_scratch_space(module, rank, size)
    }

    pub fn encrypt_pk_scratch_space(module: &Module<FFT64>, rank: usize, pk_size: usize) -> usize {
        GGLWECiphertext::encrypt_pk_scratch_space(module, rank, pk_size)
    }

    pub fn keyswitch_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        in_size: usize,
        ksk_size: usize,
        rank: usize,
    ) -> usize {
        GLWESwitchingKey::keyswitch_scratch_space(module, out_size, rank, in_size, rank, ksk_size)
    }

    pub fn keyswitch_inplace_scratch_space(module: &Module<FFT64>, out_size: usize, out_rank: usize, ksk_size: usize) -> usize {
        GLWESwitchingKey::keyswitch_inplace_scratch_space(module, out_size, out_rank, ksk_size)
    }

    pub fn automorphism_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        in_size: usize,
        ksk_size: usize,
        rank: usize,
    ) -> usize {
        let tmp_dft: usize = module.bytes_of_vec_znx_dft(rank + 1, in_size);
        let tmp_idft: usize = module.bytes_of_vec_znx_big(rank + 1, out_size);
        let idft: usize = module.vec_znx_idft_tmp_bytes();
        let keyswitch: usize = GLWECiphertext::keyswitch_inplace_scratch_space(module, out_size, rank, ksk_size);
        tmp_dft + tmp_idft + idft + keyswitch
    }

    pub fn external_product_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        in_size: usize,
        ggsw_size: usize,
        rank: usize,
    ) -> usize {
        GLWESwitchingKey::external_product_scratch_space(module, out_size, in_size, ggsw_size, rank)
    }

    pub fn external_product_inplace_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        ggsw_size: usize,
        rank: usize,
    ) -> usize {
        GLWESwitchingKey::external_product_inplace_scratch_space(module, out_size, ggsw_size, rank)
    }
}

impl<DataSelf> AutomorphismKey<DataSelf, FFT64>
where
    MatZnxDft<DataSelf, FFT64>: MatZnxDftToMut<FFT64> + MatZnxDftToRef<FFT64>,
{
    pub fn encrypt_sk<DataSk>(
        &mut self,
        module: &Module<FFT64>,
        p: i64,
        sk: &SecretKey<DataSk>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        ScalarZnx<DataSk>: ScalarZnxToRef,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), module.n());
            assert_eq!(sk.n(), module.n());
            assert_eq!(self.rank_out(), self.rank_in());
            assert_eq!(sk.rank(), self.rank());
        }

        let (sk_out_dft_data, scratch_1) = scratch.tmp_scalar_znx_dft(module, sk.rank());

        let mut sk_out_dft: SecretKeyFourier<&mut [u8], FFT64> = SecretKeyFourier {
            data: sk_out_dft_data,
            dist: sk.dist,
        };

        {
            (0..self.rank()).for_each(|i| {
                let (mut sk_inv_auto, _) = scratch_1.tmp_scalar_znx(module, 1);
                module.scalar_znx_automorphism(module.galois_element_inv(p), &mut sk_inv_auto, 0, sk, i);
                module.svp_prepare(&mut sk_out_dft, i, &sk_inv_auto, 0);
            });
        }

        self.key.encrypt_sk(
            module,
            &sk,
            &sk_out_dft,
            source_xa,
            source_xe,
            sigma,
            scratch_1,
        );

        self.p = p;
    }
}

impl<DataSelf> AutomorphismKey<DataSelf, FFT64>
where
    MatZnxDft<DataSelf, FFT64>: MatZnxDftToMut<FFT64> + MatZnxDftToRef<FFT64>,
{
    pub fn automorphism<DataLhs, DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &AutomorphismKey<DataLhs, FFT64>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut base2k::Scratch,
    ) where
        MatZnxDft<DataLhs, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.rank_in(),
                lhs.rank_in(),
                "ksk_out input rank: {} != ksk_in input rank: {}",
                self.rank_in(),
                lhs.rank_in()
            );
            assert_eq!(
                lhs.rank_out(),
                rhs.rank_in(),
                "ksk_in output rank: {} != ksk_apply input rank: {}",
                self.rank_out(),
                rhs.rank_in()
            );
            assert_eq!(
                self.rank_out(),
                rhs.rank_out(),
                "ksk_out output rank: {} != ksk_apply output rank: {}",
                self.rank_out(),
                rhs.rank_out()
            );
        }

        let cols_out: usize = rhs.rank_out() + 1;

        let (tmp_dft_data, scratch1) = scratch.tmp_vec_znx_dft(module, cols_out, lhs.size());

        let mut tmp_dft: GLWECiphertextFourier<&mut [u8], FFT64> = GLWECiphertextFourier::<&mut [u8], FFT64> {
            data: tmp_dft_data,
            basek: lhs.basek(),
            k: lhs.k(),
        };

        (0..self.rank_in()).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                // Extracts relevant row
                lhs.get_row(module, row_j, col_i, &mut tmp_dft);

                // Get a VecZnxBig from scratch space
                let (mut tmp_idft_data, scratch2) = scratch1.tmp_vec_znx_big(module, cols_out, self.size());

                // Switches input outside of DFT
                (0..cols_out).for_each(|i| {
                    module.vec_znx_idft(&mut tmp_idft_data, i, &tmp_dft.data, i, scratch2);
                });

                // Consumes to small vec znx
                let mut tmp_idft_small_data: VecZnx<&mut [u8]> = tmp_idft_data.to_vec_znx_small();

                // Reverts the automorphis key from (-pi^{-1}_{k}(s)a + s, a) to (-sa + pi_{k}(s), a)
                (0..cols_out).for_each(|i| {
                    module.vec_znx_automorphism_inplace(lhs.p(), &mut tmp_idft_small_data, i);
                });

                // Wraps into ciphertext
                let mut tmp_idft: GLWECiphertext<&mut [u8]> = GLWECiphertext::<&mut [u8]> {
                    data: tmp_idft_small_data,
                    basek: self.basek(),
                    k: self.k(),
                };

                // Key-switch (-sa + pi_{k}(s), a) to (-pi^{-1}_{k'}(s)a + pi_{k}(s), a)
                tmp_idft.keyswitch_inplace(module, &rhs.key, scratch2);

                // Applies back the automorphism X^{k}: (-pi^{-1}_{k'}(s)a + pi_{k}(s), a) -> (-pi^{-1}_{k'+k}(s)a + s, a)
                // and switches back to DFT domain
                (0..self.rank_out() + 1).for_each(|i| {
                    module.vec_znx_automorphism_inplace(lhs.p(), &mut tmp_idft, i);
                    module.vec_znx_dft(&mut tmp_dft, i, &tmp_idft, i);
                });

                // Sets back the relevant row
                self.set_row(module, row_j, col_i, &tmp_dft);
            });
        });

        tmp_dft.data.zero();

        (self.rows().min(lhs.rows())..self.rows()).for_each(|row_i| {
            (0..self.rank_in()).for_each(|col_j| {
                self.set_row(module, row_i, col_j, &tmp_dft);
            });
        });

        self.p = (lhs.p * rhs.p) % (module.cyclotomic_order() as i64);
    }

    pub fn keyswitch<DataLhs, DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &AutomorphismKey<DataLhs, FFT64>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut base2k::Scratch,
    ) where
        MatZnxDft<DataLhs, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        self.key.keyswitch(module, &lhs.key, rhs, scratch);
    }

    pub fn keyswitch_inplace<DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut base2k::Scratch,
    ) where
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        self.key.keyswitch_inplace(module, &rhs.key, scratch);
    }

    pub fn external_product<DataLhs, DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &AutomorphismKey<DataLhs, FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataLhs, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        self.key.external_product(module, &lhs.key, rhs, scratch);
    }

    pub fn external_product_inplace<DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        self.key.external_product_inplace(module, rhs, scratch);
    }
}
