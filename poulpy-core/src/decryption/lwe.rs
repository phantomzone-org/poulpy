use poulpy_hal::{
    api::{VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, HostBackend, HostDataMut, Module, ScratchArena, VecZnxReborrowBackendRef, ZnxView, ZnxViewMut},
};

pub use crate::api::LWEDecrypt;
use crate::{
    ScratchArenaTakeCore,
    layouts::{
        LWE, LWEInfos, LWEPlaintext, LWEPlaintextToBackendMut, LWEPlaintextToMut, LWESecret, LWESecretToRef, LWEToRef,
        SetLWEInfos,
    },
};

pub(crate) trait LWEDecryptDefault<BE: Backend>: Sized + VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes {
    fn lwe_decrypt_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos,
    {
        let lvl_0: usize = LWEPlaintext::bytes_of(infos.size());
        let lvl_1: usize = self.vec_znx_normalize_tmp_bytes();

        lvl_0 + lvl_1
    }

    fn lwe_decrypt_default<'s, R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: LWEToRef,
        P: LWEPlaintextToMut + LWEPlaintextToBackendMut<BE> + SetLWEInfos + LWEInfos,
        S: LWESecretToRef,
        BE: HostBackend,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        let res: &LWE<&[u8]> = &res.to_ref();
        let sk: LWESecret<&[u8]> = sk.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), sk.n());
        }
        assert!(
            scratch.available() >= self.lwe_decrypt_tmp_bytes_default(res),
            "scratch.available(): {} < LWEDecrypt::lwe_decrypt_tmp_bytes: {}",
            scratch.available(),
            self.lwe_decrypt_tmp_bytes_default(res)
        );

        let scratch = scratch.borrow();

        let (mut tmp, mut scratch_1) = scratch.take_lwe_plaintext(res);
        for i in 0..res.size() {
            tmp.data.at_mut(0, i)[0] = res.data.at(0, i)[0]
                + res.data.at(0, i)[1..]
                    .iter()
                    .zip(sk.data.at(0, 0))
                    .map(|(x, y)| x * y)
                    .sum::<i64>();
        }

        let pt_base2k = pt.base2k().into();
        let res_base2k = res.base2k().into();
        let mut pt = pt.to_backend_mut();
        let tmp_ref =
            <poulpy_hal::layouts::VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&tmp.data);
        self.vec_znx_normalize(&mut pt.data, pt_base2k, 0, 0, &tmp_ref, res_base2k, 0, &mut scratch_1);
    }
}

impl<BE: Backend> LWEDecryptDefault<BE> for Module<BE> where Self: Sized + VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes {}
