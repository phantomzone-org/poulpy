#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    api::{ModuleN, VecZnxAddScalarAssignBackend, VecZnxNormalizeInplaceBackend},
    layouts::{
        Backend, HostDataMut, Module, ScalarZnxToBackendRef, ScratchArena, VecZnxReborrowBackendMut, VecZnxReborrowBackendRef,
        ZnxZero,
    },
    source::Source,
};

use crate::{
    EncryptionInfos, GGSWNoise, ScratchArenaTakeCore,
    encryption::{GGSWEncryptSk, GLWEEncryptSkInternal},
    layouts::{
        GGSWCompressedSeedMut, GGSWInfos, GLWEPlaintext, LWEInfos,
        compressed::{GGSWCompressedToBackendMut, ggsw_compressed_at_backend_mut_from_mut},
        prepared::GLWESecretPreparedToBackendRef,
    },
};

#[doc(hidden)]
pub trait GGSWCompressedEncryptSkDefault<BE: Backend> {
    fn ggsw_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_compressed_encrypt_sk<'s, R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGSWCompressedToBackendMut<BE> + GGSWCompressedSeedMut + GGSWInfos,
        P: ScalarZnxToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut;
}

impl<BE: Backend> GGSWCompressedEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN
        + GLWEEncryptSkInternal<BE>
        + GGSWEncryptSk<BE>
        + GGSWNoise<BE>
        + VecZnxAddScalarAssignBackend<BE>
        + VecZnxNormalizeInplaceBackend<BE>,
{
    fn ggsw_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        assert_eq!(self.n() as u32, infos.n());
        let lvl_0: usize = self.ggsw_encrypt_sk_tmp_bytes(infos);
        lvl_0
    }

    fn ggsw_compressed_encrypt_sk<'s, R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGSWCompressedToBackendMut<BE> + GGSWCompressedSeedMut + GGSWInfos,
        P: ScalarZnxToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        let base2k: usize = res.base2k().into();
        let rank: usize = res.rank().into();
        let cols: usize = rank + 1;
        let dsize: usize = res.dsize().into();

        let sk_ref = sk.to_backend_ref();
        let pt_backend = pt.to_backend_ref();

        assert_eq!(res.rank(), sk_ref.rank());
        assert_eq!(pt_backend.n, self.n());
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(sk_ref.n(), self.n() as u32);
        assert!(
            scratch.available()
                >= <Module<BE> as GGSWCompressedEncryptSkDefault<BE>>::ggsw_compressed_encrypt_sk_tmp_bytes(self, res),
            "scratch.available(): {} < GGSWCompressedEncryptSk::ggsw_compressed_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            <Module<BE> as GGSWCompressedEncryptSkDefault<BE>>::ggsw_compressed_encrypt_sk_tmp_bytes(self, res)
        );

        let mut seeds: Vec<[u8; 32]> = vec![[0u8; 32]; res.dnum().as_usize() * (res.rank().as_usize() + 1)];

        {
            let mut res = res.to_backend_mut();

            let scratch = scratch.borrow();
            let (mut tmp_pt, mut scratch_1) = scratch.take_glwe_plaintext(&res);

            let mut source = Source::new(seed_xa);

            for row_i in 0..res.dnum().into() {
                tmp_pt.data.zero();

                // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
                {
                    let mut tmp_pt_data =
                        <poulpy_hal::layouts::VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendMut<BE>>::reborrow_backend_mut(
                            &mut tmp_pt.data,
                        );
                    self.vec_znx_add_scalar_assign_backend(&mut tmp_pt_data, 0, (dsize - 1) + row_i * dsize, &pt_backend, 0);
                }
                scratch_1 = scratch_1.apply_mut(|scratch| {
                    let mut tmp_pt_data =
                        <poulpy_hal::layouts::VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendMut<BE>>::reborrow_backend_mut(
                            &mut tmp_pt.data,
                        );
                    self.vec_znx_normalize_inplace_backend(base2k, &mut tmp_pt_data, 0, scratch)
                });

                for col_j in 0..rank + 1 {
                    // rlwe encrypt of vec_znx_pt into vec_znx_ct

                    let (seed, mut source_xa_tmp) = source.branch();

                    seeds[row_i * cols + col_j] = seed;

                    let tmp_pt_backend = GLWEPlaintext {
                        data: <poulpy_hal::layouts::VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(
                            &tmp_pt.data,
                        ),
                        base2k: tmp_pt.base2k,
                    };
                    let base2k = res.base2k().into();
                    let mut ct = ggsw_compressed_at_backend_mut_from_mut::<BE>(&mut res, row_i, col_j);
                    self.glwe_encrypt_sk_internal(
                        base2k,
                        &mut ct.data,
                        cols,
                        true,
                        Some((tmp_pt_backend, col_j)),
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
