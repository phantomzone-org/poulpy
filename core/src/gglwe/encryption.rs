use backend::{
    FFT64, Module, ScalarZnx, ScalarZnxAlloc, ScalarZnxDftOps, ScalarZnxOps, Scratch, VecZnxAlloc, VecZnxDftAlloc, VecZnxOps,
    ZnxInfos, ZnxView, ZnxViewMut, ZnxZero,
};
use sampling::source::Source;

use crate::{
    FourierGLWESecret, GGLWECiphertext, GLWEAutomorphismKey, GLWECiphertext, GLWESecret, GLWESwitchingKey, GLWETensorKey, Infos,
    ScratchCore, SetRow,
};

impl GGLWECiphertext<Vec<u8>, FFT64> {
    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, basek: usize, k: usize, rank: usize) -> usize {
        let size = k.div_ceil(basek);
        GLWECiphertext::encrypt_sk_scratch_space(module, basek, k)
            + module.bytes_of_vec_znx(rank + 1, size)
            + module.bytes_of_vec_znx(1, size)
            + module.bytes_of_vec_znx_dft(rank + 1, size)
    }

    pub fn encrypt_pk_scratch_space(_module: &Module<FFT64>, _basek: usize, _k: usize, _rank: usize) -> usize {
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
            assert_eq!(
                self.rank_in(),
                pt.cols(),
                "self.rank_in(): {} != pt.cols(): {}",
                self.rank_in(),
                pt.cols()
            );
            assert_eq!(
                self.rank_out(),
                sk.rank(),
                "self.rank_out(): {} != sk.rank(): {}",
                self.rank_out(),
                sk.rank()
            );
            assert_eq!(self.n(), module.n());
            assert_eq!(sk.n(), module.n());
            assert_eq!(pt.n(), module.n());
            assert!(
                scratch.available() >= GGLWECiphertext::encrypt_sk_scratch_space(module, self.basek(), self.k(), self.rank()),
                "scratch.available: {} < GGLWECiphertext::encrypt_sk_scratch_space(module, self.rank()={}, self.size()={}): {}",
                scratch.available(),
                self.rank(),
                self.size(),
                GGLWECiphertext::encrypt_sk_scratch_space(module, self.basek(), self.k(), self.rank())
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
        let (mut tmp_ct_dft, scratch_3) = scrach_2.tmp_fourier_glwe_ct(module, basek, k, rank_out);

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
    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, basek: usize, k: usize, rank_in: usize, rank_out: usize) -> usize {
        GGLWECiphertext::encrypt_sk_scratch_space(module, basek, k, rank_out)
            + module.bytes_of_scalar_znx(rank_in)
            + FourierGLWESecret::bytes_of(module, rank_out)
    }

    pub fn encrypt_pk_scratch_space(
        module: &Module<FFT64>,
        _basek: usize,
        _k: usize,
        _rank_in: usize,
        _rank_out: usize,
    ) -> usize {
        GGLWECiphertext::encrypt_pk_scratch_space(module, _basek, _k, _rank_out)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GLWESwitchingKey<DataSelf, FFT64> {
    pub fn encrypt_sk<DataSkIn: AsRef<[u8]>, DataSkOut: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        sk_in: &GLWESecret<DataSkIn>,
        sk_out: &FourierGLWESecret<DataSkOut, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) {
        #[cfg(debug_assertions)]
        {
            assert!(sk_in.n() <= module.n());
            assert!(sk_out.n() <= module.n());
        }

        let (mut sk_in_tmp, scratch1) = scratch.tmp_scalar_znx(module, sk_in.rank());
        sk_in_tmp.zero();

        (0..sk_in.rank()).for_each(|i| {
            sk_in_tmp
                .at_mut(i, 0)
                .iter_mut()
                .step_by(module.n() / sk_in.n())
                .zip(sk_in.data.at(i, 0).iter())
                .for_each(|(x, y)| *x = *y);
        });

        let (mut sk_out_tmp, scratch2) = scratch1.tmp_fourier_glwe_secret(module, sk_out.rank());
        (0..sk_out.rank()).for_each(|i| {
            sk_out_tmp
                .data
                .at_mut(i, 0)
                .chunks_exact_mut(module.n() / sk_out.n())
                .zip(sk_out.data.at(i, 0).iter())
                .for_each(|(a_chunk, &b_elem)| {
                    a_chunk.fill(b_elem);
                });
        });

        self.key.encrypt_sk(
            module,
            &sk_in_tmp,
            &sk_out_tmp,
            source_xa,
            source_xe,
            sigma,
            scratch2,
        );
        self.sk_in_n = sk_in.n();
        self.sk_out_n = sk_out.n();
    }
}

impl GLWEAutomorphismKey<Vec<u8>, FFT64> {
    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, basek: usize, k: usize, rank: usize) -> usize {
        GLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k, rank, rank) + GLWESecret::bytes_of(module, rank)
    }

    pub fn encrypt_pk_scratch_space(module: &Module<FFT64>, _basek: usize, _k: usize, _rank: usize) -> usize {
        GLWESwitchingKey::encrypt_pk_scratch_space(module, _basek, _k, _rank, _rank)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GLWEAutomorphismKey<DataSelf, FFT64> {
    pub fn encrypt_sk<DataSk: AsRef<[u8]>>(
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
                scratch.available() >= GLWEAutomorphismKey::encrypt_sk_scratch_space(module, self.basek(), self.k(), self.rank()),
                "scratch.available(): {} < AutomorphismKey::encrypt_sk_scratch_space(module, self.rank()={}, self.size()={}): {}",
                scratch.available(),
                self.rank(),
                self.size(),
                GLWEAutomorphismKey::encrypt_sk_scratch_space(module, self.basek(), self.k(), self.rank())
            )
        }

        let (mut sk_out_dft, scratch_1) = scratch.tmp_fourier_glwe_secret(module, sk.rank());

        {
            let (mut sk_out, _) = scratch_1.tmp_glwe_secret(module, sk.rank());
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

impl GLWETensorKey<Vec<u8>, FFT64> {
    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, basek: usize, k: usize, rank: usize) -> usize {
        GLWESecret::bytes_of(module, 1)
            + FourierGLWESecret::bytes_of(module, 1)
            + GLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k, rank, rank)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GLWETensorKey<DataSelf, FFT64> {
    pub fn encrypt_sk<DataSk: AsRef<[u8]>>(
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

        let (mut sk_ij, scratch1) = scratch.tmp_glwe_secret(module, 1);
        let (mut sk_ij_dft, scratch2) = scratch1.tmp_fourier_glwe_secret(module, 1);

        (0..rank).for_each(|i| {
            (i..rank).for_each(|j| {
                module.svp_apply(&mut sk_ij_dft.data, 0, &sk.data, i, &sk.data, j);
                module.scalar_znx_idft(&mut sk_ij.data, 0, &sk_ij_dft.data, 0, scratch2);
                self.at_mut(i, j)
                    .encrypt_sk(module, &sk_ij, sk, source_xa, source_xe, sigma, scratch2);
            });
        })
    }
}
