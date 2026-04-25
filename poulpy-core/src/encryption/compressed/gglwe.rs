#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    api::{ModuleN, VecZnxAddScalarAssignBackend, VecZnxDftBytesOf, VecZnxNormalizeInplaceBackend, VecZnxNormalizeTmpBytes},
    layouts::{
        Backend, HostDataMut, Module, ScalarZnxToBackendRef, ScratchArena, VecZnxReborrowBackendMut, VecZnxReborrowBackendRef,
        ZnxZero,
    },
    source::Source,
};

use crate::{
    EncryptionInfos, ScratchArenaTakeCore,
    encryption::{GLWEEncryptSk, GLWEEncryptSkInternal},
    layouts::{
        GGLWECompressedSeedMut, GGLWEInfos, GLWEPlaintext, GLWEPlaintextToRef, LWEInfos,
        compressed::{GGLWECompressedToBackendMut, gglwe_compressed_at_backend_mut_from_mut},
        prepared::GLWESecretPreparedToBackendRef,
    },
};

#[doc(hidden)]
pub trait GGLWECompressedEncryptSkDefault<BE: Backend> {
    fn gglwe_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_compressed_encrypt_sk<'s, R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut,
        P: ScalarZnxToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut;
}

impl<BE: Backend> GGLWECompressedEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN
        + GLWEEncryptSkInternal<BE>
        + GLWEEncryptSk<BE>
        + VecZnxDftBytesOf
        + VecZnxNormalizeInplaceBackend<BE>
        + VecZnxAddScalarAssignBackend<BE>
        + VecZnxNormalizeTmpBytes,
{
    fn gglwe_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(infos);
        let lvl_1: usize = self.glwe_encrypt_sk_tmp_bytes(infos).max(self.vec_znx_normalize_tmp_bytes());

        lvl_0 + lvl_1
    }

    #[allow(clippy::too_many_arguments)]
    fn gglwe_compressed_encrypt_sk<'s, R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut,
        P: ScalarZnxToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        let mut seeds: Vec<[u8; 32]> = vec![[0u8; 32]; res.seed_mut().len()];

        {
            let mut res = res.to_backend_mut();
            let pt_backend = pt.to_backend_ref();
            let sk_ref = sk.to_backend_ref();

            assert_eq!(
                res.rank_in(),
                pt_backend.cols as u32,
                "res.rank_in(): {} != pt.cols(): {}",
                res.rank_in(),
                pt_backend.cols
            );
            assert_eq!(
                res.rank_out(),
                sk_ref.rank(),
                "res.rank_out(): {} != sk.rank(): {}",
                res.rank_out(),
                sk_ref.rank()
            );
            assert_eq!(res.n(), sk_ref.n());
            assert_eq!(pt_backend.n as u32, sk_ref.n());
            assert!(
                scratch.available()
                    >= <Module<BE> as GGLWECompressedEncryptSkDefault<BE>>::gglwe_compressed_encrypt_sk_tmp_bytes(self, &res),
                "scratch.available(): {} < GGLWECompressedEncryptSk::gglwe_compressed_encrypt_sk_tmp_bytes: {}",
                scratch.available(),
                <Module<BE> as GGLWECompressedEncryptSkDefault<BE>>::gglwe_compressed_encrypt_sk_tmp_bytes(self, &res)
            );
            assert!(
                res.dnum().0 * res.dsize().0 * res.base2k().0 <= res.max_k().0,
                "res.dnum() : {} * res.dsize() : {} * res.base2k() : {} = {} >= res.k() = {}",
                res.dnum(),
                res.dsize(),
                res.base2k(),
                res.dnum().0 * res.dsize().0 * res.base2k().0,
                res.max_k()
            );

            let dnum: usize = res.dnum().into();
            let dsize: usize = res.dsize().into();
            let base2k: usize = res.base2k().into();
            let rank_in: usize = res.rank_in().into();
            let cols: usize = (res.rank_out() + 1).into();

            let mut source_xa = Source::new(seed);

            let scratch = scratch.borrow();
            let (mut tmp_pt, mut scratch_1) = scratch.take_glwe_plaintext(&res);

            for col_j in 0..rank_in {
                for row_i in 0..dnum {
                    // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
                    tmp_pt.data.zero(); // zeroes for next iteration
                    {
                        let mut tmp_pt_data =
                            <poulpy_hal::layouts::VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendMut<BE>>::reborrow_backend_mut(
                                &mut tmp_pt.data,
                            );
                        self.vec_znx_add_scalar_assign_backend(
                            &mut tmp_pt_data,
                            0,
                            (dsize - 1) + row_i * dsize,
                            &pt_backend,
                            col_j,
                        );
                    }
                    scratch_1 = scratch_1.apply_mut(|scratch| {
                        let mut tmp_pt_data =
                            <poulpy_hal::layouts::VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendMut<BE>>::reborrow_backend_mut(
                                &mut tmp_pt.data,
                            );
                        self.vec_znx_normalize_inplace_backend(base2k, &mut tmp_pt_data, 0, scratch)
                    });

                    let (seed, mut source_xa_tmp) = source_xa.branch();
                    seeds[row_i * rank_in + col_j] = seed;

                    let tmp_pt_ref = tmp_pt.to_ref();
                    let tmp_pt_backend = GLWEPlaintext {
                        data: <poulpy_hal::layouts::VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(
                            &tmp_pt.data,
                        ),
                        base2k: tmp_pt.base2k,
                    };
                    let base2k = res.base2k().into();
                    let mut ct = gglwe_compressed_at_backend_mut_from_mut::<BE>(&mut res, row_i, col_j);
                    self.glwe_encrypt_sk_internal(
                        base2k,
                        &mut ct.data,
                        cols,
                        true,
                        Some((tmp_pt_ref, tmp_pt_backend, 0)),
                        sk,
                        enc_infos,
                        source_xe,
                        &mut source_xa_tmp,
                        &mut scratch_1,
                    );
                }
            }
        };

        res.seed_mut().copy_from_slice(&seeds);
    }
}
