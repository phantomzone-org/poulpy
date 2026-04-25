use poulpy_hal::{
    api::{ScratchArenaTakeBasic, VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, HostDataMut, Module, ScratchArena, VecZnx, VecZnxReborrowBackendRef, ZnxView, ZnxViewMut, ZnxZero},
};

pub use crate::api::GLWEFromLWE;
use crate::{
    ScratchArenaTakeCore,
    keyswitching::GLWEKeyswitchDefault,
    layouts::{
        GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToMut, LWE, LWEInfos, LWEToRef, glwe_backend_ref_from_mut,
        prepared::GGLWEPreparedToBackendRef,
    },
};

pub(crate) trait GLWEFromLWEDefault<BE: Backend>:
    GLWEKeyswitchDefault<BE> + VecZnxNormalizeTmpBytes + VecZnxNormalize<BE>
where
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn glwe_from_lwe_tmp_bytes_default<R, A, K>(&self, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, glwe_infos.n());
        assert_eq!(self.n() as u32, key_infos.n());

        let lvl_0: usize = GLWE::<Vec<u8>>::bytes_of(
            self.n().into(),
            key_infos.base2k(),
            lwe_infos.max_k().max(glwe_infos.max_k()),
            1u32.into(),
        );

        let lvl_1_ks: usize = self.glwe_keyswitch_tmp_bytes_default(glwe_infos, glwe_infos, key_infos);
        let lvl_1_a_conv: usize = if lwe_infos.base2k() == key_infos.base2k() {
            0
        } else {
            VecZnx::bytes_of(self.n(), 1, lwe_infos.size()) + self.vec_znx_normalize_tmp_bytes()
        };

        let lvl_1: usize = lvl_1_ks.max(lvl_1_a_conv);

        lvl_0 + lvl_1
    }

    fn glwe_from_lwe_default<'s, R, A, K>(&self, res: &mut R, lwe: &A, ksk: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToMut + GLWEToBackendMut<BE>,
        A: LWEToRef,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        let res_infos = {
            let res = res.to_mut();
            GLWELayout {
                n: res.n(),
                base2k: res.base2k(),
                k: res.max_k(),
                rank: res.rank(),
            }
        };
        let lwe: &LWE<&[u8]> = &lwe.to_ref();

        assert_eq!(res_infos.n.as_u32(), self.n() as u32);
        assert_eq!(ksk.n(), self.n() as u32);
        assert!(lwe.n() <= self.n() as u32);
        assert!(
            scratch.available() >= self.glwe_from_lwe_tmp_bytes_default(&res_infos, lwe, ksk),
            "scratch.available(): {} < GLWEFromLWE::glwe_from_lwe_tmp_bytes: {}",
            scratch.available(),
            self.glwe_from_lwe_tmp_bytes_default(&res_infos, lwe, ksk)
        );

        let scratch = scratch.borrow();

        let (mut glwe, scratch_1) = scratch.take_glwe(&GLWELayout {
            n: ksk.n(),
            base2k: ksk.base2k(),
            k: lwe.max_k(),
            rank: 1u32.into(),
        });
        glwe.data.zero();

        let n_lwe: usize = lwe.n().into();

        let mut scratch_1 = if lwe.base2k() == ksk.base2k() {
            for i in 0..lwe.size() {
                let data_lwe: &[i64] = lwe.data.at(0, i);
                glwe.data.at_mut(0, i)[0] = data_lwe[0];
                glwe.data.at_mut(1, i)[..n_lwe].copy_from_slice(&data_lwe[1..]);
            }
            scratch_1
        } else {
            {
                let (mut a_conv, mut scratch_2) = scratch_1.take_vec_znx(self.n(), 1, lwe.size());
                a_conv.zero();
                for j in 0..lwe.size() {
                    let data_lwe: &[i64] = lwe.data.at(0, j);
                    a_conv.at_mut(0, j)[0] = data_lwe[0]
                }

                self.vec_znx_normalize(
                    &mut glwe.data,
                    ksk.base2k().into(),
                    0,
                    0,
                    &<VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&a_conv),
                    lwe.base2k().into(),
                    0,
                    &mut scratch_2.borrow(),
                );

                a_conv.zero();
                for j in 0..lwe.size() {
                    let data_lwe: &[i64] = lwe.data.at(0, j);
                    a_conv.at_mut(0, j)[..n_lwe].copy_from_slice(&data_lwe[1..]);
                }

                self.vec_znx_normalize(
                    &mut glwe.data,
                    ksk.base2k().into(),
                    0,
                    1,
                    &<VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&a_conv),
                    lwe.base2k().into(),
                    0,
                    &mut scratch_2.borrow(),
                );

                scratch_2
            }
        };

        let mut res_backend = res.to_backend_mut();
        let glwe_ref = glwe_backend_ref_from_mut::<BE>(&glwe);
        self.glwe_keyswitch_default(&mut res_backend, &glwe_ref, ksk, &mut scratch_1)
    }
}

impl<BE: Backend> GLWEFromLWEDefault<BE> for Module<BE>
where
    Self: GLWEKeyswitchDefault<BE> + VecZnxNormalizeTmpBytes + VecZnxNormalize<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    for<'s> BE::BufMut<'s>: HostDataMut,
{
}
