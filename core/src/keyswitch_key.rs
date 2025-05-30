use backend::{Backend, FFT64, MatZnxDft, MatZnxDftOps, Module, Scratch, ZnxZero};
use sampling::source::Source;

use crate::{GGLWECiphertext, GGSWCiphertext, GLWECiphertextFourier, GLWESecret, GetRow, Infos, ScratchCore, SetRow};

pub struct GLWESwitchingKey<Data, B: Backend>(pub(crate) GGLWECiphertext<Data, B>);

impl GLWESwitchingKey<Vec<u8>, FFT64> {
    pub fn alloc(module: &Module<FFT64>, basek: usize, k: usize, rows: usize, rank_in: usize, rank_out: usize) -> Self {
        GLWESwitchingKey(GGLWECiphertext::alloc(
            module, basek, k, rows, rank_in, rank_out,
        ))
    }

    pub fn bytes_of(module: &Module<FFT64>, basek: usize, k: usize, rows: usize, rank_in: usize, rank_out: usize) -> usize {
        GGLWECiphertext::<Vec<u8>, FFT64>::bytes_of(module, basek, k, rows, rank_in, rank_out)
    }
}

impl<T, B: Backend> Infos for GLWESwitchingKey<T, B> {
    type Inner = MatZnxDft<T, B>;

    fn inner(&self) -> &Self::Inner {
        self.0.inner()
    }

    fn basek(&self) -> usize {
        self.0.basek()
    }

    fn k(&self) -> usize {
        self.0.k()
    }
}

impl<T, B: Backend> GLWESwitchingKey<T, B> {
    pub fn rank(&self) -> usize {
        self.0.data.cols_out() - 1
    }

    pub fn rank_in(&self) -> usize {
        self.0.data.cols_in()
    }

    pub fn rank_out(&self) -> usize {
        self.0.data.cols_out() - 1
    }
}

impl<C: AsRef<[u8]>> GetRow<FFT64> for GLWESwitchingKey<C, FFT64> {
    fn get_row<R: AsMut<[u8]> + AsRef<[u8]>>(
        &self,
        module: &Module<FFT64>,
        row_i: usize,
        col_j: usize,
        res: &mut GLWECiphertextFourier<R, FFT64>,
    ) {
        module.vmp_extract_row(&mut res.data, &self.0.data, row_i, col_j);
    }
}

impl<C: AsMut<[u8]> + AsRef<[u8]>> SetRow<FFT64> for GLWESwitchingKey<C, FFT64> {
    fn set_row<R: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        row_i: usize,
        col_j: usize,
        a: &GLWECiphertextFourier<R, FFT64>,
    ) {
        module.vmp_prepare_row(&mut self.0.data, row_i, col_j, &a.data);
    }
}

impl GLWESwitchingKey<Vec<u8>, FFT64> {
    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, basek: usize, k: usize, rank: usize) -> usize {
        GGLWECiphertext::generate_from_sk_scratch_space(module, basek, k, rank)
    }

    pub fn encrypt_pk_scratch_space(module: &Module<FFT64>, _basek: usize, _k: usize, _rank: usize) -> usize {
        GGLWECiphertext::generate_from_pk_scratch_space(module, _basek, _k, _rank)
    }

    pub fn keyswitch_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        out_k: usize,
        out_rank: usize,
        in_k: usize,
        in_rank: usize,
        ksk_k: usize,
    ) -> usize {
        let tmp_in: usize = GLWECiphertextFourier::bytes_of(module, basek, in_k, in_rank);
        let tmp_out: usize = GLWECiphertextFourier::bytes_of(module, basek, out_k, out_rank);
        let ksk: usize = GLWECiphertextFourier::keyswitch_scratch_space(module, basek, out_k, out_rank, in_k, in_rank, ksk_k);
        tmp_in + tmp_out + ksk
    }

    pub fn keyswitch_inplace_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        out_k: usize,
        out_rank: usize,
        ksk_k: usize,
    ) -> usize {
        let tmp: usize = GLWECiphertextFourier::bytes_of(module, basek, out_k, out_rank);
        let ksk: usize = GLWECiphertextFourier::keyswitch_inplace_scratch_space(module, basek, out_k, out_rank, ksk_k);
        tmp + ksk
    }

    pub fn external_product_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        out_k: usize,
        in_k: usize,
        ggsw_k: usize,
        rank: usize,
    ) -> usize {
        let tmp_in: usize = GLWECiphertextFourier::bytes_of(module, basek, in_k, rank);
        let tmp_out: usize = GLWECiphertextFourier::bytes_of(module, basek, out_k, rank);
        let ggsw: usize = GLWECiphertextFourier::external_product_scratch_space(module, basek, out_k, in_k, ggsw_k, rank);
        tmp_in + tmp_out + ggsw
    }

    pub fn external_product_inplace_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        out_k: usize,
        ggsw_k: usize,
        rank: usize,
    ) -> usize {
        let tmp: usize = GLWECiphertextFourier::bytes_of(module, basek, out_k, rank);
        let ggsw: usize = GLWECiphertextFourier::external_product_inplace_scratch_space(module, basek, out_k, ggsw_k, rank);
        tmp + ggsw
    }
}
impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GLWESwitchingKey<DataSelf, FFT64> {
    pub fn generate_from_sk<DataSkIn: AsRef<[u8]>, DataSkOut: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        sk_in: &GLWESecret<DataSkIn, FFT64>,
        sk_out: &GLWESecret<DataSkOut, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) {
        self.0.encrypt_sk(
            module,
            &sk_in.data,
            sk_out,
            source_xa,
            source_xe,
            sigma,
            scratch,
        );
    }

    pub fn keyswitch<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWESwitchingKey<DataLhs, FFT64>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
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

        let (mut tmp_in, scratch1) = scratch.tmp_glwe_fourier(module, lhs.basek(), lhs.k(), lhs.rank());
        let (mut tmp_out, scratch2) = scratch1.tmp_glwe_fourier(module, self.basek(), self.k(), self.rank());

        (0..self.rank_in()).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                lhs.get_row(module, row_j, col_i, &mut tmp_in);
                tmp_out.keyswitch(module, &tmp_in, rhs, scratch2);
                self.set_row(module, row_j, col_i, &tmp_out);
            });
        });

        tmp_out.data.zero();

        (self.rows().min(lhs.rows())..self.rows()).for_each(|row_i| {
            (0..self.rank_in()).for_each(|col_j| {
                self.set_row(module, row_i, col_j, &tmp_out);
            });
        });
    }

    pub fn keyswitch_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.rank_out(),
                rhs.rank_out(),
                "ksk_out output rank: {} != ksk_apply output rank: {}",
                self.rank_out(),
                rhs.rank_out()
            );
        }

        let (mut tmp, scratch1) = scratch.tmp_glwe_fourier(module, self.basek(), self.k(), self.rank());

        (0..self.rank_in()).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                self.get_row(module, row_j, col_i, &mut tmp);
                tmp.keyswitch_inplace(module, rhs, scratch1);
                self.set_row(module, row_j, col_i, &tmp);
            });
        });
    }

    pub fn external_product<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWESwitchingKey<DataLhs, FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
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
                rhs.rank(),
                "ksk_in output rank: {} != ggsw rank: {}",
                self.rank_out(),
                rhs.rank()
            );
            assert_eq!(
                self.rank_out(),
                rhs.rank(),
                "ksk_out output rank: {} != ggsw rank: {}",
                self.rank_out(),
                rhs.rank()
            );
        }

        let (mut tmp_in, scratch1) = scratch.tmp_glwe_fourier(module, lhs.basek(), lhs.k(), lhs.rank());
        let (mut tmp_out, scratch2) = scratch1.tmp_glwe_fourier(module, self.basek(), self.k(), self.rank());

        (0..self.rank_in()).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                lhs.get_row(module, row_j, col_i, &mut tmp_in);
                tmp_out.external_product(module, &tmp_in, rhs, scratch2);
                self.set_row(module, row_j, col_i, &tmp_out);
            });
        });

        tmp_out.data.zero();

        (self.rows().min(lhs.rows())..self.rows()).for_each(|row_i| {
            (0..self.rank_in()).for_each(|col_j| {
                self.set_row(module, row_i, col_j, &tmp_out);
            });
        });
    }

    pub fn external_product_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.rank_out(),
                rhs.rank(),
                "ksk_out output rank: {} != ggsw rank: {}",
                self.rank_out(),
                rhs.rank()
            );
        }

        let (mut tmp, scratch1) = scratch.tmp_glwe_fourier(module, self.basek(), self.k(), self.rank());

        (0..self.rank_in()).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                self.get_row(module, row_j, col_i, &mut tmp);
                tmp.external_product_inplace(module, rhs, scratch1);
                self.set_row(module, row_j, col_i, &tmp);
            });
        });
    }
}
