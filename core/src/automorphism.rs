use backend::{Backend, FFT64, MatZnxDft, MatZnxDftOps, Module, ScalarZnxOps, Scratch, VecZnx, VecZnxDftOps, VecZnxOps, ZnxZero};
use sampling::source::Source;

use crate::{
    GGLWECiphertext, GGSWCiphertext, GLWECiphertext, GLWECiphertextFourier, GLWESecret, GLWESwitchingKey, GetRow, Infos,
    ScratchCore, SetRow,
};

pub struct AutomorphismKey<Data, B: Backend> {
    pub(crate) key: GLWESwitchingKey<Data, B>,
    pub(crate) p: i64,
}

impl AutomorphismKey<Vec<u8>, FFT64> {
    pub fn alloc(module: &Module<FFT64>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self {
        AutomorphismKey {
            key: GLWESwitchingKey::alloc(module, basek, k, rows, digits, rank, rank),
            p: 0,
        }
    }

    pub fn bytes_of(module: &Module<FFT64>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize {
        GLWESwitchingKey::<Vec<u8>, FFT64>::bytes_of(module, basek, k, rows, digits,rank, rank)
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

    pub fn digits(&self) -> usize {
        self.key.digits()
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

impl<C: AsRef<[u8]>> GetRow<FFT64> for AutomorphismKey<C, FFT64> {
    fn get_row<R: AsMut<[u8]> + AsRef<[u8]>>(
        &self,
        module: &Module<FFT64>,
        row_i: usize,
        col_j: usize,
        res: &mut GLWECiphertextFourier<R, FFT64>,
    ) {
        module.vmp_extract_row(&mut res.data, &self.key.0.data, row_i, col_j);
    }
}

impl<C: AsMut<[u8]> + AsRef<[u8]>> SetRow<FFT64> for AutomorphismKey<C, FFT64> {
    fn set_row<R: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        row_i: usize,
        col_j: usize,
        a: &GLWECiphertextFourier<R, FFT64>,
    ) {
        module.vmp_prepare_row(&mut self.key.0.data, row_i, col_j, &a.data);
    }
}

impl AutomorphismKey<Vec<u8>, FFT64> {
    pub fn generate_from_sk_scratch_space(module: &Module<FFT64>, basek: usize, k: usize, rank: usize) -> usize {
        GGLWECiphertext::generate_from_sk_scratch_space(module, basek, k, rank) + GLWESecret::bytes_of(module, rank)
    }

    pub fn generate_from_pk_scratch_space(module: &Module<FFT64>, _basek: usize, _k: usize, _rank: usize) -> usize {
        GGLWECiphertext::generate_from_pk_scratch_space(module, _basek, _k, _rank)
    }

    pub fn keyswitch_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        out_k: usize,
        in_k: usize,
        ksk_k: usize,
        rank: usize,
    ) -> usize {
        GLWESwitchingKey::keyswitch_scratch_space(module, basek, out_k, rank, in_k, rank, ksk_k)
    }

    pub fn keyswitch_inplace_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        out_k: usize,
        out_rank: usize,
        ksk_k: usize,
    ) -> usize {
        GLWESwitchingKey::keyswitch_inplace_scratch_space(module, basek, out_k, out_rank, ksk_k)
    }

    pub fn automorphism_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        out_k: usize,
        in_k: usize,
        atk_k: usize,
        rank: usize,
    ) -> usize {
        let tmp_dft: usize = GLWECiphertextFourier::bytes_of(module, basek, in_k, rank);
        let tmp_idft: usize = GLWECiphertextFourier::bytes_of(module, basek, out_k, rank);
        let idft: usize = module.vec_znx_idft_tmp_bytes();
        let keyswitch: usize = GLWECiphertext::keyswitch_inplace_scratch_space(module, basek, out_k, rank, atk_k);
        tmp_dft + tmp_idft + idft + keyswitch
    }

    pub fn automorphism_inplace_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        out_k: usize,
        ksk_k: usize,
        rank: usize,
    ) -> usize {
        AutomorphismKey::automorphism_scratch_space(module, basek, out_k, out_k, ksk_k, rank)
    }

    pub fn external_product_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        out_k: usize,
        in_k: usize,
        ggsw_k: usize,
        rank: usize,
    ) -> usize {
        GLWESwitchingKey::external_product_scratch_space(module, basek, out_k, in_k, ggsw_k, rank)
    }

    pub fn external_product_inplace_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        out_k: usize,
        ggsw_k: usize,
        rank: usize,
    ) -> usize {
        GLWESwitchingKey::external_product_inplace_scratch_space(module, basek, out_k, ggsw_k, rank)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> AutomorphismKey<DataSelf, FFT64> {
    pub fn generate_from_sk<DataSk: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        p: i64,
        sk: &GLWESecret<DataSk, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), module.n());
            assert_eq!(sk.n(), module.n());
            assert_eq!(self.rank_out(), self.rank_in());
            assert_eq!(sk.rank(), self.rank());
            assert!(
                scratch.available()
                    >= AutomorphismKey::generate_from_sk_scratch_space(module, self.basek(), self.k(), self.rank()),
                "scratch.available(): {} < AutomorphismKey::generate_from_sk_scratch_space(module, self.rank()={}, \
                 self.size()={}): {}",
                scratch.available(),
                self.rank(),
                self.size(),
                AutomorphismKey::generate_from_sk_scratch_space(module, self.basek(), self.k(), self.rank())
            )
        }

        let (mut sk_out, scratch_1) = scratch.tmp_sk(module, sk.rank());
        (0..self.rank()).for_each(|i| {
            module.scalar_znx_automorphism(
                module.galois_element_inv(p),
                &mut sk_out.data,
                i,
                &sk.data,
                i,
            );
        });

        sk_out.prep_fourier(module);

        self.key
            .generate_from_sk(module, &sk, &sk_out, source_xa, source_xe, sigma, scratch_1);

        self.p = p;
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> AutomorphismKey<DataSelf, FFT64> {
    pub fn automorphism<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &AutomorphismKey<DataLhs, FFT64>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
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

        let (mut tmp_dft, scratch1) = scratch.tmp_glwe_fourier(module, lhs.basek(), lhs.k(), lhs.rank());

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
                    module.vec_znx_automorphism_inplace(lhs.p(), &mut tmp_idft.data, i);
                    module.vec_znx_dft(&mut tmp_dft.data, i, &tmp_idft.data, i);
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

    pub fn automorphism_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        unsafe {
            let self_ptr: *mut AutomorphismKey<DataSelf, FFT64> = self as *mut AutomorphismKey<DataSelf, FFT64>;
            self.automorphism(&module, &*self_ptr, rhs, scratch);
        }
    }

    pub fn keyswitch<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &AutomorphismKey<DataLhs, FFT64>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        self.key.keyswitch(module, &lhs.key, rhs, scratch);
    }

    pub fn keyswitch_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        self.key.keyswitch_inplace(module, &rhs.key, scratch);
    }

    pub fn external_product<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &AutomorphismKey<DataLhs, FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        self.key.external_product(module, &lhs.key, rhs, scratch);
    }

    pub fn external_product_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        self.key.external_product_inplace(module, rhs, scratch);
    }
}
