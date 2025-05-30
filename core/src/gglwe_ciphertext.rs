use backend::{
    Backend, FFT64, MatZnxDft, MatZnxDftAlloc, MatZnxDftOps, Module, ScalarZnx, Scratch, VecZnxAlloc, VecZnxDftAlloc, VecZnxOps,
    ZnxInfos, ZnxZero,
};
use sampling::source::Source;

use crate::{GLWECiphertext, GLWECiphertextFourier, GLWESecret, GetRow, Infos, ScratchCore, SetRow, derive_size};

pub struct GGLWECiphertext<C, B: Backend> {
    pub(crate) data: MatZnxDft<C, B>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
}

impl<B: Backend> GGLWECiphertext<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rows: usize, rank_in: usize, rank_out: usize) -> Self {
        Self {
            data: module.new_mat_znx_dft(rows, rank_in, rank_out + 1, derive_size(basek, k)),
            basek: basek,
            k,
        }
    }

    pub fn bytes_of(module: &Module<FFT64>, basek: usize, k: usize, rows: usize, rank_in: usize, rank_out: usize) -> usize {
        module.bytes_of_mat_znx_dft(rows, rank_in, rank_out + 1, derive_size(basek, k))
    }
}

impl<T, B: Backend> Infos for GGLWECiphertext<T, B> {
    type Inner = MatZnxDft<T, B>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.basek
    }

    fn k(&self) -> usize {
        self.k
    }
}

impl<T, B: Backend> GGLWECiphertext<T, B> {
    pub fn rank(&self) -> usize {
        self.data.cols_out() - 1
    }

    pub fn rank_in(&self) -> usize {
        self.data.cols_in()
    }

    pub fn rank_out(&self) -> usize {
        self.data.cols_out() - 1
    }
}

impl GGLWECiphertext<Vec<u8>, FFT64> {
    pub fn generate_from_sk_scratch_space(module: &Module<FFT64>, basek: usize, k: usize, rank: usize) -> usize {
        let size = derive_size(basek, k);
        GLWECiphertext::encrypt_sk_scratch_space(module, basek, k)
            + module.bytes_of_vec_znx(rank + 1, size)
            + module.bytes_of_vec_znx(1, size)
            + module.bytes_of_vec_znx_dft(rank + 1, size)
    }

    pub fn generate_from_pk_scratch_space(_module: &Module<FFT64>, _basek: usize, _k: usize, _rank: usize) -> usize {
        unimplemented!()
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GGLWECiphertext<DataSelf, FFT64> {
    pub fn encrypt_sk<DataPt: AsRef<[u8]>, DataSk: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        pt: &ScalarZnx<DataPt>,
        sk: &GLWESecret<DataSk, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank_in(), pt.cols());
            assert_eq!(self.rank_out(), sk.rank());
            assert_eq!(self.n(), module.n());
            assert_eq!(sk.n(), module.n());
            assert_eq!(pt.n(), module.n());
            assert!(
                scratch.available()
                    >= GGLWECiphertext::generate_from_sk_scratch_space(module, self.basek(), self.k(), self.rank()),
                "scratch.available: {} < GGLWECiphertext::generate_from_sk_scratch_space(module, self.rank()={}, \
                 self.size()={}): {}",
                scratch.available(),
                self.rank(),
                self.size(),
                GGLWECiphertext::generate_from_sk_scratch_space(module, self.basek(), self.k(), self.rank())
            )
        }

        let rows: usize = self.rows();
        let basek: usize = self.basek();
        let k: usize = self.k();
        let rank_in: usize = self.rank_in();
        let rank_out: usize = self.rank_out();

        let (mut tmp_pt, scrach_1) = scratch.tmp_glwe_pt(module, basek, k);
        let (mut tmp_ct, scrach_2) = scrach_1.tmp_glwe_ct(module, basek, k, rank_out);
        let (mut tmp_ct_dft, scratch_3) = scrach_2.tmp_glwe_fourier(module, basek, k, rank_out);

        // For each input column (i.e. rank) produces a GGLWE ciphertext of rank_out+1 columns
        //
        // Example for ksk rank 2 to rank 3:
        //
        // (-(a0*s0 + a1*s1 + a2*s2) + s0', a0, a1, a2)
        // (-(b0*s0 + b1*s1 + b2*s2) + s0', b0, b1, b2)
        //
        // Example ksk rank 2 to rank 1
        //
        // (-(a*s) + s0, a)
        // (-(b*s) + s1, b)
        (0..rank_in).for_each(|col_i| {
            (0..rows).for_each(|row_i| {
                // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
                tmp_pt.data.zero(); // zeroes for next iteration
                module.vec_znx_add_scalar_inplace(&mut tmp_pt.data, 0, row_i, pt, col_i); // Selects the i-th
                module.vec_znx_normalize_inplace(basek, &mut tmp_pt.data, 0, scratch_3);

                // rlwe encrypt of vec_znx_pt into vec_znx_ct
                tmp_ct.encrypt_sk(module, &tmp_pt, sk, source_xa, source_xe, sigma, scratch_3);

                // Switch vec_znx_ct into DFT domain
                tmp_ct.dft(module, &mut tmp_ct_dft);

                // Stores vec_znx_dft_ct into thw i-th row of the MatZnxDft
                self.set_row(module, row_i, col_i, &tmp_ct_dft);
            });
        });
    }
}

impl<C: AsRef<[u8]>> GetRow<FFT64> for GGLWECiphertext<C, FFT64> {
    fn get_row<R: AsMut<[u8]> + AsRef<[u8]>>(
        &self,
        module: &Module<FFT64>,
        row_i: usize,
        col_j: usize,
        res: &mut GLWECiphertextFourier<R, FFT64>,
    ) {
        module.vmp_extract_row(&mut res.data, &self.data, row_i, col_j);
    }
}

impl<C: AsMut<[u8]> + AsRef<[u8]>> SetRow<FFT64> for GGLWECiphertext<C, FFT64> {
    fn set_row<R: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        row_i: usize,
        col_j: usize,
        a: &GLWECiphertextFourier<R, FFT64>,
    ) {
        module.vmp_prepare_row(&mut self.data, row_i, col_j, &a.data);
    }
}
