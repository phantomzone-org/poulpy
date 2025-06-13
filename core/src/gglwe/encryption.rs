use backend::{
    FFT64, Module, ScalarZnx, ScalarZnxDftOps, ScalarZnxOps, Scratch, VecZnxAlloc, VecZnxDftAlloc, VecZnxOps, ZnxInfos, ZnxZero,
};
use sampling::source::Source;

use crate::{
    FourierGLWESecret, GGLWECiphertext, GLWEAutomorphismKey, GLWECiphertext, GLWESecret, GLWESwitchingKey, GLWETensorKey, Infos,
    ScratchCore, SetRow,
};

impl GGLWECiphertext<Vec<u8>, FFT64> {
    pub fn generate_from_sk_scratch_space(module: &Module<FFT64>, basek: usize, k: usize, rank: usize) -> usize {
        let size = k.div_ceil(basek);
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
        sk: &FourierGLWESecret<DataSk, FFT64>,
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
            );
            assert!(
                self.rows() * self.digits() * self.basek() <= self.k(),
                "self.rows() : {} * self.digits() : {} * self.basek() : {} = {} >= self.k() = {}",
                self.rows(),
                self.digits(),
                self.basek(),
                self.rows() * self.digits() * self.basek(),
                self.k()
            );
        }

        let rows: usize = self.rows();
        let digits: usize = self.digits();
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
                module.vec_znx_add_scalar_inplace(
                    &mut tmp_pt.data,
                    0,
                    (digits - 1) + row_i * digits,
                    pt,
                    col_i,
                );
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

impl GLWESwitchingKey<Vec<u8>, FFT64> {
    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, basek: usize, k: usize, rank: usize) -> usize {
        GGLWECiphertext::generate_from_sk_scratch_space(module, basek, k, rank)
    }

    pub fn encrypt_pk_scratch_space(module: &Module<FFT64>, _basek: usize, _k: usize, _rank: usize) -> usize {
        GGLWECiphertext::generate_from_pk_scratch_space(module, _basek, _k, _rank)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GLWESwitchingKey<DataSelf, FFT64> {
    pub fn generate_from_sk<DataSkIn: AsRef<[u8]>, DataSkOut: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        sk_in: &GLWESecret<DataSkIn>,
        sk_out: &FourierGLWESecret<DataSkOut, FFT64>,
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
}

impl GLWEAutomorphismKey<Vec<u8>, FFT64> {
    pub fn generate_from_sk_scratch_space(module: &Module<FFT64>, basek: usize, k: usize, rank: usize) -> usize {
        GGLWECiphertext::generate_from_sk_scratch_space(module, basek, k, rank) + GLWESecret::bytes_of(module, rank)
    }

    pub fn generate_from_pk_scratch_space(module: &Module<FFT64>, _basek: usize, _k: usize, _rank: usize) -> usize {
        GGLWECiphertext::generate_from_pk_scratch_space(module, _basek, _k, _rank)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GLWEAutomorphismKey<DataSelf, FFT64> {
    pub fn generate_from_sk<DataSk: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        p: i64,
        sk: &GLWESecret<DataSk>,
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
                    >= GLWEAutomorphismKey::generate_from_sk_scratch_space(module, self.basek(), self.k(), self.rank()),
                "scratch.available(): {} < AutomorphismKey::generate_from_sk_scratch_space(module, self.rank()={}, \
                 self.size()={}): {}",
                scratch.available(),
                self.rank(),
                self.size(),
                GLWEAutomorphismKey::generate_from_sk_scratch_space(module, self.basek(), self.k(), self.rank())
            )
        }

        let (mut sk_out_dft, scratch_1) = scratch.tmp_fourier_sk(module, sk.rank());

        {
            let (mut sk_out, _) = scratch_1.tmp_sk(module, sk.rank());
            (0..self.rank()).for_each(|i| {
                module.scalar_znx_automorphism(
                    module.galois_element_inv(p),
                    &mut sk_out.data,
                    i,
                    &sk.data,
                    i,
                );
            });
            sk_out_dft.set(module, &sk_out);
        }

        self.key.generate_from_sk(
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

impl GLWETensorKey<Vec<u8>, FFT64> {
    pub fn generate_from_sk_scratch_space(module: &Module<FFT64>, basek: usize, k: usize, rank: usize) -> usize {
        GLWESecret::bytes_of(module, 1)
            + FourierGLWESecret::bytes_of(module, 1)
            + GLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k, rank)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GLWETensorKey<DataSelf, FFT64> {
    pub fn generate_from_sk<DataSk: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        sk: &FourierGLWESecret<DataSk, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), sk.rank());
            assert_eq!(self.n(), module.n());
            assert_eq!(sk.n(), module.n());
        }

        let rank: usize = self.rank();

        let (mut sk_ij, scratch1) = scratch.tmp_sk(module, 1);
        let (mut sk_ij_dft, scratch2) = scratch1.tmp_fourier_sk(module, 1);

        (0..rank).for_each(|i| {
            (i..rank).for_each(|j| {
                module.svp_apply(&mut sk_ij_dft.data, 0, &sk.data, i, &sk.data, j);
                module.scalar_znx_idft(&mut sk_ij.data, 0, &sk_ij_dft.data, 0, scratch2);
                self.at_mut(i, j)
                    .generate_from_sk(module, &sk_ij, sk, source_xa, source_xe, sigma, scratch2);
            });
        })
    }
}
