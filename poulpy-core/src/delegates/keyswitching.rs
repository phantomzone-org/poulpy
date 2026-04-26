use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    api::{GGLWEKeyswitch, GGSWKeyswitch, GLWEKeyswitch, LWEKeySwitch},
    keyswitching::{GGLWEKeyswitchDefault, GGSWKeyswitchDefault, LWEKeySwitchDefault},
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWEBackendMut,
        GLWEBackendRef, GLWEInfos, LWEInfos, LWEToBackendMut, LWEToBackendRef,
        prepared::{GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
    },
    oep::{GGLWEKeyswitchImpl, GGSWKeyswitchImpl, GLWEKeyswitchImpl, LWEKeyswitchImpl},
};

macro_rules! impl_keyswitching_delegate {
    ($trait:ty, [$($bounds:tt)+], $($body:item)+) => {
        impl<BE> $trait for Module<BE>
        where
            $($bounds)+
        {
            $($body)+
        }
    };
}

impl_keyswitching_delegate!(
    GLWEKeyswitch<BE>,
    [BE: Backend + GLWEKeyswitchImpl<BE>],
    fn glwe_keyswitch_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, key_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos,
    {
        BE::glwe_keyswitch_tmp_bytes(self, res_infos, a_infos, key_infos)
    }

    fn glwe_keyswitch<'s, 'r, 'a, K>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &GLWEBackendRef<'a, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    )
    where
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        BE::glwe_keyswitch(self, res, a, key, scratch)
    }

    fn glwe_keyswitch_inplace<'s, 'r, K>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    )
    where
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        BE::glwe_keyswitch_assign(self, res, key, scratch)
    }
);

impl_keyswitching_delegate!(
    GGLWEKeyswitch<BE>,
    [BE: Backend + GGLWEKeyswitchImpl<BE>, Module<BE>: GGLWEKeyswitchDefault<BE>],
    fn gglwe_keyswitch_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
    {
        BE::gglwe_keyswitch_tmp_bytes(self, res_infos, a_infos, key_infos)
    }

    fn gglwe_keyswitch<'s, R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        B: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        BE::gglwe_keyswitch(self, res, a, b, scratch)
    }

    fn gglwe_keyswitch_inplace<'s, R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        BE::gglwe_keyswitch_assign(self, res, a, scratch)
    }
);

impl_keyswitching_delegate!(
    GGSWKeyswitch<BE>,
    [BE: Backend + GGSWKeyswitchImpl<BE>, Module<BE>: GGSWKeyswitchDefault<BE>],
    fn ggsw_keyswitch_tmp_bytes<R, A, K, T>(&self, res_infos: &R, a_infos: &A, key_infos: &K, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        BE::ggsw_keyswitch_tmp_bytes(self, res_infos, a_infos, key_infos, tsk_infos)
    }

    fn ggsw_keyswitch<'s, R, A, K, T>(
        &self,
        res: &mut R,
        a: &A,
        key: &K,
        tsk: &T,
        scratch: &mut ScratchArena<'s, BE>,
    )
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        BE::ggsw_keyswitch(self, res, a, key, tsk, scratch)
    }

    fn ggsw_keyswitch_inplace<'s, R, K, T>(
        &self,
        res: &mut R,
        key: &K,
        tsk: &T,
        scratch: &mut ScratchArena<'s, BE>,
    )
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        BE::ggsw_keyswitch_assign(self, res, key, tsk, scratch)
    }
);

impl_keyswitching_delegate!(
    LWEKeySwitch<BE>,
    [BE: Backend + LWEKeyswitchImpl<BE>, Module<BE>: LWEKeySwitchDefault<BE>],
    fn lwe_keyswitch_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        BE::lwe_keyswitch_tmp_bytes(self, res_infos, a_infos, key_infos)
    }

    fn lwe_keyswitch<'s, R, A, K>(&self, res: &mut R, a: &A, ksk: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
        for<'x> <BE as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
    {
        BE::lwe_keyswitch(self, res, a, ksk, scratch)
    }
);
