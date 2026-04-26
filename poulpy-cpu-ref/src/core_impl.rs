use crate::{FFT64Ref, NTT120Ref};
use poulpy_core::{
    ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendRef, GGLWEToBackendMut, GGLWEToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef,
        GGSWInfos, GGSWPreparedToBackendRef, GGSWToMut, GLWEInfos, GLWEPlaintext,
        GLWESecretPrepared, GLWESecretTensorPrepared, GLWETensor, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        LWEPlaintextToMut, LWESecretToRef, LWEToRef, SetLWEInfos,
    },
    oep::{
        AutomorphismDefaults, AutomorphismImpl, ConversionDefaults, ConversionImpl, DecryptionDefaults, DecryptionImpl,
        GGLWEExternalProductDefaults, GGLWEExternalProductImpl, GGLWEKeyswitchDefaults, GGLWEKeyswitchImpl,
        GGSWExternalProductDefaults, GGSWExternalProductImpl, GGSWKeyswitchDefaults, GGSWKeyswitchImpl, GGSWRotateImpl,
        GLWEExternalProductDefaults, GLWEExternalProductImpl, GLWEKeyswitchDefaults, GLWEKeyswitchImpl, GLWEMulConstImpl,
        GLWEMulPlainImpl, GLWEMulXpMinusOneImpl, GLWENormalizeImpl, GLWEPackImpl, GLWERotateImpl, GLWEShiftImpl,
        GLWETensoringImpl, GLWETraceImpl, LWEKeyswitchDefaults, LWEKeyswitchImpl, OperationsDefaults,
    },
};
use poulpy_hal::layouts::{Backend, HostBackend, HostDataMut, HostDataRef, Module, ScratchArena};

macro_rules! decryption_helper {
    ($(fn $name:ident [$($gen:tt)*] ($($args:tt)*) -> $ret:ty where [$($bounds:tt)*] => $method:ident($($call:tt)*);)+) => {
        $(
            pub fn $name<$($gen)*>($($args)*) -> $ret
            where
                $($bounds)*
            {
                <BE as DecryptionDefaults<BE>>::$method($($call)*)
            }
        )+
    };
}

decryption_helper! {
    fn decryption_glwe_decrypt_tmp_bytes [BE, A] (module: &Module<BE>, infos: &A) -> usize
    where [BE: Backend + DecryptionDefaults<BE>, A: GLWEInfos,] => glwe_decrypt_tmp_bytes(module, infos);

    fn decryption_glwe_decrypt ['s, BE, R, P, S] (
        module: &Module<BE>,
        res: &R,
        pt: &mut P,
        sk: &S,
        scratch: &mut ScratchArena<'s, BE>
    ) -> ()
    where [
        BE: Backend + DecryptionDefaults<BE> + HostBackend,
        R: poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        P: poulpy_core::layouts::GLWEPlaintextToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        S: poulpy_core::layouts::prepared::GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    ] => glwe_decrypt(module, res, pt, sk, scratch);

    fn decryption_lwe_decrypt_tmp_bytes [BE, A] (module: &Module<BE>, infos: &A) -> usize
    where [BE: Backend + DecryptionDefaults<BE>, A: LWEInfos,] => lwe_decrypt_tmp_bytes(module, infos);

    fn decryption_lwe_decrypt [BE, R, P, S] (
        module: &Module<BE>,
        res: &R,
        pt: &mut P,
        sk: &S,
        scratch: &mut ScratchArena<'_, BE>
    ) -> ()
    where [
        BE: Backend + DecryptionDefaults<BE> + HostBackend,
        R: LWEToRef,
        P: LWEPlaintextToMut + poulpy_core::layouts::LWEPlaintextToBackendMut<BE> + SetLWEInfos + LWEInfos,
        S: LWESecretToRef,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    ] => lwe_decrypt(module, res, pt, sk, scratch);

    fn decryption_glwe_tensor_decrypt_tmp_bytes [BE, A] (module: &Module<BE>, infos: &A) -> usize
    where [BE: Backend + DecryptionDefaults<BE>, A: GLWEInfos,] => glwe_tensor_decrypt_tmp_bytes(module, infos);

    fn decryption_glwe_tensor_decrypt [BE, R, P, S0, S1] (
        module: &Module<BE>,
        res: &GLWETensor<R>,
        pt: &mut GLWEPlaintext<P>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut ScratchArena<'_, BE>
    ) -> ()
    where [
        BE: Backend + DecryptionDefaults<BE> + HostBackend,
        R: HostDataRef,
        GLWETensor<R>: poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        P: HostDataMut,
        GLWEPlaintext<P>: poulpy_core::layouts::GLWEPlaintextToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        S0: HostDataRef,
        S1: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut,
    ] => glwe_tensor_decrypt(module, res, pt, sk, sk_tensor, scratch);
}

macro_rules! conversion_helper {
    () => {};
    (fn $name:ident [$($gen:tt)*] ($($args:tt)*) -> $ret:ty where [$($bounds:tt)*] => $method:ident($($call:tt)*); $($rest:tt)*) => {
        pub fn $name<$($gen)*>($($args)*) -> $ret
        where
            $($bounds)*
        {
            <BE as ConversionDefaults<BE>>::$method($($call)*)
        }
        conversion_helper! { $($rest)* }
    };
    (fn $name:ident [$($gen:tt)*] ($($args:tt)*) where [$($bounds:tt)*] => $method:ident($($call:tt)*); $($rest:tt)*) => {
        pub fn $name<$($gen)*>($($args)*)
        where
            $($bounds)*
        {
            <BE as ConversionDefaults<BE>>::$method($($call)*)
        }
        conversion_helper! { $($rest)* }
    };
}

conversion_helper! {
    fn conversion_glwe_from_lwe_tmp_bytes [BE, R, A, K] (
        module: &Module<BE>,
        glwe_infos: &R,
        lwe_infos: &A,
        key_infos: &K
    ) -> usize
    where [BE: Backend + ConversionDefaults<BE>, R: GLWEInfos, A: LWEInfos, K: GGLWEInfos,]
    => glwe_from_lwe_tmp_bytes(module, glwe_infos, lwe_infos, key_infos);

    fn conversion_glwe_from_lwe ['s, BE, R, A, K] (
        module: &Module<BE>,
        res: &mut R,
        lwe: &A,
        ksk: &K,
        scratch: &mut ScratchArena<'s, BE>
    )
    where [
        BE: Backend + ConversionDefaults<BE>,
        R: poulpy_core::layouts::GLWEToBackendMut<BE> + GLWEInfos,
        A: poulpy_core::layouts::LWEToBackendRef<BE> + LWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        BE: 's,
    ]
    => glwe_from_lwe(module, res, lwe, ksk, scratch);

    fn conversion_lwe_from_glwe_tmp_bytes [BE, R, A, K] (
        module: &Module<BE>,
        lwe_infos: &R,
        glwe_infos: &A,
        key_infos: &K
    ) -> usize
    where [BE: Backend + ConversionDefaults<BE>, R: LWEInfos, A: GLWEInfos, K: GGLWEInfos,]
    => lwe_from_glwe_tmp_bytes(module, lwe_infos, glwe_infos, key_infos);

    fn conversion_lwe_from_glwe ['s, BE, R, A, K] (
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        a_idx: usize,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>
    )
    where [
        BE: Backend + ConversionDefaults<BE>,
        R: poulpy_core::layouts::LWEToBackendMut<BE> + LWEInfos,
        A: poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        BE: 's,
    ] => lwe_from_glwe(module, res, a, a_idx, key, scratch);

    fn conversion_ggsw_from_gglwe_tmp_bytes [BE, R, A] (module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where [BE: Backend + ConversionDefaults<BE>, R: GGSWInfos, A: GGLWEInfos,]
    => ggsw_from_gglwe_tmp_bytes(module, res_infos, tsk_infos);

    fn conversion_ggsw_from_gglwe ['s, BE, R, A, T] (
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'s, BE>
    )
    where [
        BE: Backend + ConversionDefaults<BE>,
        R: GGSWToMut + poulpy_core::layouts::GGSWToBackendMut<BE> + GGSWInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        BE: 's,
    ] => ggsw_from_gglwe(module, res, a, tsk, scratch);

    fn conversion_ggsw_expand_rows_tmp_bytes [BE, R, A] (module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where [BE: Backend + ConversionDefaults<BE>, R: GGSWInfos, A: GGLWEInfos,]
    => ggsw_expand_rows_tmp_bytes(module, res_infos, tsk_infos);

    fn conversion_ggsw_expand_row ['s, 'r, BE, T] (
        module: &Module<BE>,
        res: &mut poulpy_core::layouts::GGSWBackendMut<'r, BE>,
        tsk: &T,
        scratch: &mut ScratchArena<'s, BE>
    )
    where [
        BE: Backend + ConversionDefaults<BE>,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        BE: 's,
    ] => ggsw_expand_row(module, res, tsk, scratch);
}

macro_rules! external_helper {
    () => {};
    (fn $name:ident [$($gen:tt)*] ($($args:tt)*) -> $ret:ty where [$($bounds:tt)*] => [$defaults:path]::$method:ident($($call:tt)*); $($rest:tt)*) => {
        pub fn $name<$($gen)*>($($args)*) -> $ret
        where
            $($bounds)*
        {
            <BE as $defaults>::$method($($call)*)
        }
        external_helper! { $($rest)* }
    };
    (fn $name:ident [$($gen:tt)*] ($($args:tt)*) where [$($bounds:tt)*] => [$defaults:path]::$method:ident($($call:tt)*); $($rest:tt)*) => {
        pub fn $name<$($gen)*>($($args)*)
        where
            $($bounds)*
        {
            <BE as $defaults>::$method($($call)*)
        }
        external_helper! { $($rest)* }
    };
}

external_helper! {
    fn external_glwe_external_product_tmp_bytes [BE, R, A, G] (
        module: &Module<BE>,
        res_infos: &R,
        a_infos: &A,
        ggsw_infos: &G
    ) -> usize
    where [BE: Backend + GLWEExternalProductDefaults<BE>, R: GLWEInfos, A: GLWEInfos, G: GGSWInfos,]
    => [GLWEExternalProductDefaults<BE>]::glwe_external_product_tmp_bytes(module, res_infos, a_infos, ggsw_infos);

    fn external_glwe_external_product ['s, 'r, 'a, BE, G] (
        module: &Module<BE>,
        res: &mut poulpy_core::layouts::GLWEBackendMut<'r, BE>,
        a: &poulpy_core::layouts::GLWEBackendRef<'a, BE>,
        ggsw: &G,
        scratch: &mut ScratchArena<'s, BE>
    )
    where [
        BE: GLWEExternalProductDefaults<BE>,
        G: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    ] => [GLWEExternalProductDefaults<BE>]::glwe_external_product(module, res, a, ggsw, scratch);

    fn external_glwe_external_product_inplace ['s, 'r, BE, G] (
        module: &Module<BE>,
        res: &mut poulpy_core::layouts::GLWEBackendMut<'r, BE>,
        ggsw: &G,
        scratch: &mut ScratchArena<'s, BE>
    )
    where [
        BE: Backend + GLWEExternalProductDefaults<BE>,
        G: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    ] => [GLWEExternalProductDefaults<BE>]::glwe_external_product_inplace(module, res, ggsw, scratch);

    fn external_gglwe_external_product_tmp_bytes [BE, R, A, B] (
        module: &Module<BE>,
        res_infos: &R,
        a_infos: &A,
        b_infos: &B
    ) -> usize
    where [BE: Backend + GGLWEExternalProductDefaults<BE>, R: GGLWEInfos, A: GGLWEInfos, B: GGSWInfos,]
    => [GGLWEExternalProductDefaults<BE>]::gglwe_external_product_tmp_bytes(module, res_infos, a_infos, b_infos);

    fn external_gglwe_external_product ['s, BE, R, A, B] (
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        b: &B,
        scratch: &mut ScratchArena<'s, BE>
    )
    where [
        BE: Backend + GGLWEExternalProductDefaults<BE>,
        R: poulpy_core::layouts::GGLWEToBackendMut<BE> + GGLWEInfos,
        A: poulpy_core::layouts::GGLWEToBackendRef<BE> + GGLWEInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    ] => [GGLWEExternalProductDefaults<BE>]::gglwe_external_product(module, res, a, b, scratch);

    fn external_gglwe_external_product_inplace ['s, BE, R, A] (
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        scratch: &mut ScratchArena<'s, BE>
    )
    where [
        BE: Backend + GGLWEExternalProductDefaults<BE>,
        R: poulpy_core::layouts::GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    ] => [GGLWEExternalProductDefaults<BE>]::gglwe_external_product_inplace(module, res, a, scratch);

    fn external_ggsw_external_product_tmp_bytes [BE, R, A, B] (
        module: &Module<BE>,
        res_infos: &R,
        a_infos: &A,
        b_infos: &B
    ) -> usize
    where [BE: Backend + GGSWExternalProductDefaults<BE>, R: GGSWInfos, A: GGSWInfos, B: GGSWInfos,]
    => [GGSWExternalProductDefaults<BE>]::ggsw_external_product_tmp_bytes(module, res_infos, a_infos, b_infos);

    fn external_ggsw_external_product ['s, BE, R, A, B] (
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        b: &B,
        scratch: &mut ScratchArena<'s, BE>
    )
    where [
        BE: Backend + GGSWExternalProductDefaults<BE>,
        R: poulpy_core::layouts::GGSWToBackendMut<BE> + GGSWInfos,
        A: poulpy_core::layouts::GGSWToBackendRef<BE> + GGSWInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    ] => [GGSWExternalProductDefaults<BE>]::ggsw_external_product(module, res, a, b, scratch);

    fn external_ggsw_external_product_inplace ['s, BE, R, A] (
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        scratch: &mut ScratchArena<'s, BE>
    )
    where [
        BE: Backend + GGSWExternalProductDefaults<BE>,
        R: poulpy_core::layouts::GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    ] => [GGSWExternalProductDefaults<BE>]::ggsw_external_product_inplace(module, res, a, scratch);
}

macro_rules! keyswitch_helper {
    () => {};
    (fn $name:ident [$($gen:tt)*] ($($args:tt)*) -> $ret:ty where [$($bounds:tt)*] => [$defaults:path]::$method:ident($($call:tt)*); $($rest:tt)*) => {
        pub fn $name<$($gen)*>($($args)*) -> $ret
        where
            $($bounds)*
        {
            <BE as $defaults>::$method($($call)*)
        }
        keyswitch_helper! { $($rest)* }
    };
    (fn $name:ident [$($gen:tt)*] ($($args:tt)*) where [$($bounds:tt)*] => [$defaults:path]::$method:ident($($call:tt)*); $($rest:tt)*) => {
        pub fn $name<$($gen)*>($($args)*)
        where
            $($bounds)*
        {
            <BE as $defaults>::$method($($call)*)
        }
        keyswitch_helper! { $($rest)* }
    };
}

keyswitch_helper! {
    fn keyswitch_glwe_keyswitch_tmp_bytes [BE, R, A, K] (
        module: &Module<BE>,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K
    ) -> usize
    where [BE: Backend + GLWEKeyswitchDefaults<BE>, R: GLWEInfos, A: GLWEInfos, K: GGLWEInfos,]
    => [GLWEKeyswitchDefaults<BE>]::glwe_keyswitch_tmp_bytes(module, res_infos, a_infos, key_infos);

    fn keyswitch_glwe_keyswitch ['s, BE, K] (
        module: &Module<BE>,
        res: &mut poulpy_core::layouts::GLWEBackendMut<'_, BE>,
        a: &poulpy_core::layouts::GLWEBackendRef<'_, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>
    )
    where [
        BE: Backend + GLWEKeyswitchDefaults<BE>,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    ] => [GLWEKeyswitchDefaults<BE>]::glwe_keyswitch(module, res, a, key, scratch);

    fn keyswitch_glwe_keyswitch_inplace ['s, BE, K] (
        module: &Module<BE>,
        res: &mut poulpy_core::layouts::GLWEBackendMut<'_, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>
    )
    where [
        BE: Backend + GLWEKeyswitchDefaults<BE>,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    ] => [GLWEKeyswitchDefaults<BE>]::glwe_keyswitch_inplace(module, res, key, scratch);

    fn keyswitch_gglwe_keyswitch_tmp_bytes [BE, R, A, K] (
        module: &Module<BE>,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K
    ) -> usize
    where [BE: Backend + GGLWEKeyswitchDefaults<BE>, R: GGLWEInfos, A: GGLWEInfos, K: GGLWEInfos,]
    => [GGLWEKeyswitchDefaults<BE>]::gglwe_keyswitch_tmp_bytes(module, res_infos, a_infos, key_infos);

    fn keyswitch_gglwe_keyswitch ['s, BE, R, A, K] (
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>
    )
    where [
        BE: Backend + GGLWEKeyswitchDefaults<BE>,
        R: poulpy_core::layouts::GGLWEToBackendMut<BE> + GGLWEInfos,
        A: poulpy_core::layouts::GGLWEToBackendRef<BE> + GGLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    ] => [GGLWEKeyswitchDefaults<BE>]::gglwe_keyswitch(module, res, a, key, scratch);

    fn keyswitch_gglwe_keyswitch_inplace ['s, BE, R, K] (
        module: &Module<BE>,
        res: &mut R,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>
    )
    where [
        BE: Backend + GGLWEKeyswitchDefaults<BE>,
        R: poulpy_core::layouts::GGLWEToBackendMut<BE> + GGLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    ] => [GGLWEKeyswitchDefaults<BE>]::gglwe_keyswitch_inplace(module, res, key, scratch);

    fn keyswitch_ggsw_keyswitch_tmp_bytes [BE, R, A, K, T] (
        module: &Module<BE>,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K,
        tsk_infos: &T
    ) -> usize
    where [BE: Backend + GGSWKeyswitchDefaults<BE>, R: GGSWInfos, A: GGSWInfos, K: GGLWEInfos, T: GGLWEInfos,]
    => [GGSWKeyswitchDefaults<BE>]::ggsw_keyswitch_tmp_bytes(module, res_infos, a_infos, key_infos, tsk_infos);

    fn keyswitch_ggsw_keyswitch ['s, BE, R, A, K, T] (
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        key: &K,
        tsk: &T,
        scratch: &mut ScratchArena<'s, BE>
    )
    where [
        BE: Backend + GGSWKeyswitchDefaults<BE>,
        R: poulpy_core::layouts::GGSWToBackendMut<BE> + GGSWInfos,
        A: poulpy_core::layouts::GGSWToBackendRef<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    ] => [GGSWKeyswitchDefaults<BE>]::ggsw_keyswitch(module, res, a, key, tsk, scratch);

    fn keyswitch_ggsw_keyswitch_inplace ['s, BE, R, K, T] (
        module: &Module<BE>,
        res: &mut R,
        key: &K,
        tsk: &T,
        scratch: &mut ScratchArena<'s, BE>
    )
    where [
        BE: Backend + GGSWKeyswitchDefaults<BE>,
        R: poulpy_core::layouts::GGSWToBackendMut<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    ] => [GGSWKeyswitchDefaults<BE>]::ggsw_keyswitch_inplace(module, res, key, tsk, scratch);

    fn keyswitch_lwe_keyswitch_tmp_bytes [BE, R, A, K] (
        module: &Module<BE>,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K
    ) -> usize
    where [BE: Backend + LWEKeyswitchDefaults<BE>, R: LWEInfos, A: LWEInfos, K: GGLWEInfos,]
    => [LWEKeyswitchDefaults<BE>]::lwe_keyswitch_tmp_bytes(module, res_infos, a_infos, key_infos);

    fn keyswitch_lwe_keyswitch ['s, BE, R, A, K] (
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        ksk: &K,
        scratch: &mut ScratchArena<'s, BE>
    )
    where [
        BE: Backend + LWEKeyswitchDefaults<BE>,
        R: poulpy_core::layouts::LWEToBackendMut<BE> + LWEInfos,
        A: poulpy_core::layouts::LWEToBackendRef<BE> + LWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    ] => [LWEKeyswitchDefaults<BE>]::lwe_keyswitch(module, res, a, ksk, scratch);
}

#[macro_export]
macro_rules! impl_decryption_via_helpers {
    ($be:ty, $($helpers:tt)*) => {
        unsafe impl DecryptionImpl<$be> for $be {
            fn glwe_decrypt_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
            where
                A: GLWEInfos,
            {
                $($helpers)*::decryption_glwe_decrypt_tmp_bytes(module, infos)
            }

            fn glwe_decrypt<'s, R, P, S>(module: &Module<Self>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, Self>)
            where
                R: poulpy_core::layouts::GLWEToBackendRef<Self> + GLWEInfos,
                P: poulpy_core::layouts::GLWEPlaintextToBackendMut<Self> + GLWEInfos + SetLWEInfos,
                S: poulpy_core::layouts::prepared::GLWESecretPreparedToBackendRef<Self> + GLWEInfos,
                Self: HostBackend + 's,
                for<'a> ScratchArena<'a, Self>: ScratchArenaTakeCore<'a, Self>,
                for<'a> <Self as Backend>::BufMut<'a>: HostDataMut,
            {
                $($helpers)*::decryption_glwe_decrypt(module, res, pt, sk, scratch)
            }

            fn lwe_decrypt_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
            where
                A: LWEInfos,
            {
                $($helpers)*::decryption_lwe_decrypt_tmp_bytes(module, infos)
            }

            fn lwe_decrypt<'s, R, P, S>(module: &Module<Self>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, Self>)
            where
                R: LWEToRef,
                P: LWEPlaintextToMut + poulpy_core::layouts::LWEPlaintextToBackendMut<Self> + SetLWEInfos + LWEInfos,
                S: LWESecretToRef,
                Self: HostBackend,
                for<'a> ScratchArena<'a, Self>: ScratchArenaTakeCore<'a, Self>,
                for<'a> <Self as Backend>::BufMut<'a>: HostDataMut,
            {
                $($helpers)*::decryption_lwe_decrypt(module, res, pt, sk, scratch)
            }

            fn glwe_tensor_decrypt<R, P, S0, S1>(
                module: &Module<Self>,
                res: &GLWETensor<R>,
                pt: &mut GLWEPlaintext<P>,
                sk: &GLWESecretPrepared<S0, Self>,
                sk_tensor: &GLWESecretTensorPrepared<S1, Self>,
                scratch: &mut ScratchArena<'_, Self>,
            ) where
                R: HostDataRef,
                GLWETensor<R>: poulpy_core::layouts::GLWEToBackendRef<Self> + GLWEInfos,
                P: HostDataMut,
                GLWEPlaintext<P>: poulpy_core::layouts::GLWEPlaintextToBackendMut<Self> + GLWEInfos + SetLWEInfos,
                S0: HostDataRef,
                S1: HostDataRef,
                Self: HostBackend,
                for<'a> <Self as Backend>::BufMut<'a>: HostDataMut,
            {
                $($helpers)*::decryption_glwe_tensor_decrypt(module, res, pt, sk, sk_tensor, scratch)
            }

            fn glwe_tensor_decrypt_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
            where
                A: GLWEInfos,
            {
                $($helpers)*::decryption_glwe_tensor_decrypt_tmp_bytes(module, infos)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_conversion_via_helpers {
    ($be:ty, $($helpers:tt)*) => {
        unsafe impl ConversionImpl<$be> for $be {
            fn glwe_from_lwe_tmp_bytes<R, A, K>(module: &Module<Self>, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
            where
                R: GLWEInfos,
                A: LWEInfos,
                K: GGLWEInfos,
            {
                $($helpers)*::conversion_glwe_from_lwe_tmp_bytes(module, glwe_infos, lwe_infos, key_infos)
            }

            fn glwe_from_lwe<'s, R, A, K>(
                module: &Module<Self>,
                res: &mut R,
                lwe: &A,
                ksk: &K,
                scratch: &mut ScratchArena<'s, Self>,
            )
            where
                R: poulpy_core::layouts::GLWEToBackendMut<Self> + GLWEInfos,
                A: poulpy_core::layouts::LWEToBackendRef<Self> + LWEInfos,
                K: GGLWEPreparedToBackendRef<Self> + GGLWEInfos,
                for<'a> ScratchArena<'a, Self>: ScratchArenaTakeCore<'a, Self>,
                for<'a> <Self as Backend>::BufMut<'a>: HostDataMut,
                Self: 's,
            {
                $($helpers)*::conversion_glwe_from_lwe(module, res, lwe, ksk, scratch)
            }

            fn lwe_from_glwe_tmp_bytes<R, A, K>(module: &Module<Self>, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
            where
                R: LWEInfos,
                A: GLWEInfos,
                K: GGLWEInfos,
            {
                $($helpers)*::conversion_lwe_from_glwe_tmp_bytes(module, lwe_infos, glwe_infos, key_infos)
            }

            fn lwe_from_glwe<'s, R, A, K>(
                module: &Module<Self>,
                res: &mut R,
                a: &A,
                a_idx: usize,
                key: &K,
                scratch: &mut ScratchArena<'s, Self>,
            )
            where
                R: poulpy_core::layouts::LWEToBackendMut<Self> + LWEInfos,
                A: poulpy_core::layouts::GLWEToBackendRef<Self> + GLWEInfos,
                K: GGLWEPreparedToBackendRef<Self> + GGLWEInfos,
                for<'a> ScratchArena<'a, Self>: ScratchArenaTakeCore<'a, Self>,
                for<'a> <Self as Backend>::BufMut<'a>: HostDataMut,
                Self: 's,
            {
                $($helpers)*::conversion_lwe_from_glwe(module, res, a, a_idx, key, scratch)
            }

            fn ggsw_from_gglwe_tmp_bytes<R, A>(module: &Module<Self>, res_infos: &R, tsk_infos: &A) -> usize
            where
                R: GGSWInfos,
                A: GGLWEInfos,
            {
                $($helpers)*::conversion_ggsw_from_gglwe_tmp_bytes(module, res_infos, tsk_infos)
            }

            fn ggsw_from_gglwe<'s, R, A, T>(
                module: &Module<Self>,
                res: &mut R,
                a: &A,
                tsk: &T,
                scratch: &mut ScratchArena<'s, Self>,
            )
            where
                R: GGSWToMut + poulpy_core::layouts::GGSWToBackendMut<Self> + GGSWInfos,
                A: GGLWEToBackendRef<Self> + GGLWEInfos,
                T: GGLWEToGGSWKeyPreparedToBackendRef<Self> + GGLWEInfos,
                for<'a> ScratchArena<'a, Self>: ScratchArenaTakeCore<'a, Self>,
                for<'a> <Self as Backend>::BufMut<'a>: HostDataMut,
                Self: 's,
            {
                $($helpers)*::conversion_ggsw_from_gglwe(module, res, a, tsk, scratch)
            }

            fn ggsw_expand_rows_tmp_bytes<R, A>(module: &Module<Self>, res_infos: &R, tsk_infos: &A) -> usize
            where
                R: GGSWInfos,
                A: GGLWEInfos,
            {
                $($helpers)*::conversion_ggsw_expand_rows_tmp_bytes(module, res_infos, tsk_infos)
            }

            fn ggsw_expand_row<'s, 'r, T>(
                module: &Module<Self>,
                res: &mut poulpy_core::layouts::GGSWBackendMut<'r, Self>,
                tsk: &T,
                scratch: &mut ScratchArena<'s, Self>,
            )
            where
                T: GGLWEToGGSWKeyPreparedToBackendRef<Self> + GGLWEInfos,
                for<'a> ScratchArena<'a, Self>: ScratchArenaTakeCore<'a, Self>,
            {
                $($helpers)*::conversion_ggsw_expand_row(module, res, tsk, scratch)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_external_product_via_helpers {
    ($be:ty, $($helpers:tt)*) => {
        unsafe impl GLWEExternalProductImpl<$be> for $be {
            fn glwe_external_product_tmp_bytes<R, A, G>(
                module: &Module<Self>,
                res_infos: &R,
                a_infos: &A,
                ggsw_infos: &G,
            ) -> usize
            where
                R: GLWEInfos,
                A: GLWEInfos,
                G: GGSWInfos,
            {
                $($helpers)*::external_glwe_external_product_tmp_bytes(module, res_infos, a_infos, ggsw_infos)
            }

            fn glwe_external_product<'s, 'r, 'a, G>(
                module: &Module<Self>,
                res: &mut poulpy_core::layouts::GLWEBackendMut<'r, Self>,
                a: &poulpy_core::layouts::GLWEBackendRef<'a, Self>,
                ggsw: &G,
                scratch: &mut ScratchArena<'s, Self>,
            )
            where
                G: GGSWPreparedToBackendRef<Self> + GGSWInfos,
                ScratchArena<'s, Self>: ScratchArenaTakeCore<'s, Self>,
                Self: 's,
            {
                $($helpers)*::external_glwe_external_product(module, res, a, ggsw, scratch)
            }

            fn glwe_external_product_inplace<'s, 'r, G>(
                module: &Module<Self>,
                res: &mut poulpy_core::layouts::GLWEBackendMut<'r, Self>,
                ggsw: &G,
                scratch: &mut ScratchArena<'s, Self>,
            )
            where
                G: GGSWPreparedToBackendRef<Self> + GGSWInfos,
                ScratchArena<'s, Self>: ScratchArenaTakeCore<'s, Self>,
                Self: 's,
            {
                $($helpers)*::external_glwe_external_product_inplace(module, res, ggsw, scratch)
            }
        }

        unsafe impl GGLWEExternalProductImpl<$be> for $be {
            fn gglwe_external_product_tmp_bytes<R, A, B>(module: &Module<Self>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
            where
                R: GGLWEInfos,
                A: GGLWEInfos,
                B: GGSWInfos,
            {
                $($helpers)*::external_gglwe_external_product_tmp_bytes(module, res_infos, a_infos, b_infos)
            }

            fn gglwe_external_product<'s, R, A, B>(
                module: &Module<Self>,
                res: &mut R,
                a: &A,
                b: &B,
                scratch: &mut ScratchArena<'s, Self>,
            )
            where
                R: poulpy_core::layouts::GGLWEToBackendMut<Self> + GGLWEInfos,
                A: poulpy_core::layouts::GGLWEToBackendRef<Self> + GGLWEInfos,
                B: GGSWPreparedToBackendRef<Self> + GGSWInfos,
                ScratchArena<'s, Self>: ScratchArenaTakeCore<'s, Self>,
                Self: 's,
            {
                $($helpers)*::external_gglwe_external_product(module, res, a, b, scratch)
            }

            fn gglwe_external_product_inplace<'s, R, A>(
                module: &Module<Self>,
                res: &mut R,
                a: &A,
                scratch: &mut ScratchArena<'s, Self>,
            )
            where
                R: poulpy_core::layouts::GGLWEToBackendMut<Self> + GGLWEInfos,
                A: GGSWPreparedToBackendRef<Self> + GGSWInfos,
                ScratchArena<'s, Self>: ScratchArenaTakeCore<'s, Self>,
                Self: 's,
            {
                $($helpers)*::external_gglwe_external_product_inplace(module, res, a, scratch)
            }
        }

        unsafe impl GGSWExternalProductImpl<$be> for $be {
            fn ggsw_external_product_tmp_bytes<R, A, B>(module: &Module<Self>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
            where
                R: GGSWInfos,
                A: GGSWInfos,
                B: GGSWInfos,
            {
                $($helpers)*::external_ggsw_external_product_tmp_bytes(module, res_infos, a_infos, b_infos)
            }

            fn ggsw_external_product<'s, R, A, B>(
                module: &Module<Self>,
                res: &mut R,
                a: &A,
                b: &B,
                scratch: &mut ScratchArena<'s, Self>,
            )
            where
                R: poulpy_core::layouts::GGSWToBackendMut<Self> + GGSWInfos,
                A: poulpy_core::layouts::GGSWToBackendRef<Self> + GGSWInfos,
                B: GGSWPreparedToBackendRef<Self> + GGSWInfos,
                ScratchArena<'s, Self>: ScratchArenaTakeCore<'s, Self>,
                Self: 's,
            {
                $($helpers)*::external_ggsw_external_product(module, res, a, b, scratch)
            }

            fn ggsw_external_product_inplace<'s, R, A>(
                module: &Module<Self>,
                res: &mut R,
                a: &A,
                scratch: &mut ScratchArena<'s, Self>,
            )
            where
                R: poulpy_core::layouts::GGSWToBackendMut<Self> + GGSWInfos,
                A: GGSWPreparedToBackendRef<Self> + GGSWInfos,
                ScratchArena<'s, Self>: ScratchArenaTakeCore<'s, Self>,
                Self: 's,
            {
                $($helpers)*::external_ggsw_external_product_inplace(module, res, a, scratch)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_keyswitching_via_helpers {
    ($be:ty, $($helpers:tt)*) => {
        unsafe impl GLWEKeyswitchImpl<$be> for $be {
            fn glwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
            where
                R: GLWEInfos,
                A: GLWEInfos,
                K: GGLWEInfos,
            {
                $($helpers)*::keyswitch_glwe_keyswitch_tmp_bytes(module, res_infos, a_infos, key_infos)
            }

            fn glwe_keyswitch<'s, K>(
                module: &Module<Self>,
                res: &mut poulpy_core::layouts::GLWEBackendMut<'_, Self>,
                a: &poulpy_core::layouts::GLWEBackendRef<'_, Self>,
                key: &K,
                scratch: &mut ScratchArena<'s, Self>,
            )
            where
                K: GGLWEPreparedToBackendRef<Self> + GGLWEInfos,
                Self: 's,
            {
                $($helpers)*::keyswitch_glwe_keyswitch(module, res, a, key, scratch)
            }

            fn glwe_keyswitch_inplace<'s, K>(
                module: &Module<Self>,
                res: &mut poulpy_core::layouts::GLWEBackendMut<'_, Self>,
                key: &K,
                scratch: &mut ScratchArena<'s, Self>,
            )
            where
                K: GGLWEPreparedToBackendRef<Self> + GGLWEInfos,
                Self: 's,
            {
                $($helpers)*::keyswitch_glwe_keyswitch_inplace(module, res, key, scratch)
            }
        }

        unsafe impl GGLWEKeyswitchImpl<$be> for $be {
            fn gglwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
            where
                R: GGLWEInfos,
                A: GGLWEInfos,
                K: GGLWEInfos,
            {
                $($helpers)*::keyswitch_gglwe_keyswitch_tmp_bytes(module, res_infos, a_infos, key_infos)
            }

            fn gglwe_keyswitch<'s, R, A, K>(
                module: &Module<Self>,
                res: &mut R,
                a: &A,
                key: &K,
                scratch: &mut ScratchArena<'s, Self>,
            )
            where
                R: poulpy_core::layouts::GGLWEToBackendMut<Self> + GGLWEInfos,
                A: poulpy_core::layouts::GGLWEToBackendRef<Self> + GGLWEInfos,
                K: GGLWEPreparedToBackendRef<Self> + GGLWEInfos,
                Self: 's,
            {
                $($helpers)*::keyswitch_gglwe_keyswitch(module, res, a, key, scratch)
            }

            fn gglwe_keyswitch_inplace<'s, R, K>(
                module: &Module<Self>,
                res: &mut R,
                key: &K,
                scratch: &mut ScratchArena<'s, Self>,
            )
            where
                R: poulpy_core::layouts::GGLWEToBackendMut<Self> + GGLWEInfos,
                K: GGLWEPreparedToBackendRef<Self> + GGLWEInfos,
                Self: 's,
            {
                $($helpers)*::keyswitch_gglwe_keyswitch_inplace(module, res, key, scratch)
            }
        }

        unsafe impl GGSWKeyswitchImpl<$be> for $be {
            fn ggsw_keyswitch_tmp_bytes<R, A, K, T>(
                module: &Module<Self>,
                res_infos: &R,
                a_infos: &A,
                key_infos: &K,
                tsk_infos: &T,
            ) -> usize
            where
                R: GGSWInfos,
                A: GGSWInfos,
                K: GGLWEInfos,
                T: GGLWEInfos,
            {
                $($helpers)*::keyswitch_ggsw_keyswitch_tmp_bytes(module, res_infos, a_infos, key_infos, tsk_infos)
            }

            fn ggsw_keyswitch<'s, R, A, K, T>(
                module: &Module<Self>,
                res: &mut R,
                a: &A,
                key: &K,
                tsk: &T,
                scratch: &mut ScratchArena<'s, Self>,
            )
            where
                R: poulpy_core::layouts::GGSWToBackendMut<Self> + GGSWInfos,
                A: poulpy_core::layouts::GGSWToBackendRef<Self> + GGSWInfos,
                K: GGLWEPreparedToBackendRef<Self> + GGLWEInfos,
                T: GGLWEToGGSWKeyPreparedToBackendRef<Self> + GGLWEInfos,
                Self: 's,
            {
                $($helpers)*::keyswitch_ggsw_keyswitch(module, res, a, key, tsk, scratch)
            }

            fn ggsw_keyswitch_inplace<'s, R, K, T>(
                module: &Module<Self>,
                res: &mut R,
                key: &K,
                tsk: &T,
                scratch: &mut ScratchArena<'s, Self>,
            )
            where
                R: poulpy_core::layouts::GGSWToBackendMut<Self> + GGSWInfos,
                K: GGLWEPreparedToBackendRef<Self> + GGLWEInfos,
                T: GGLWEToGGSWKeyPreparedToBackendRef<Self> + GGLWEInfos,
                Self: 's,
            {
                $($helpers)*::keyswitch_ggsw_keyswitch_inplace(module, res, key, tsk, scratch)
            }
        }

        unsafe impl LWEKeyswitchImpl<$be> for $be {
            fn lwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
            where
                R: LWEInfos,
                A: LWEInfos,
                K: GGLWEInfos,
            {
                $($helpers)*::keyswitch_lwe_keyswitch_tmp_bytes(module, res_infos, a_infos, key_infos)
            }

            fn lwe_keyswitch<'s, R, A, K>(
                module: &Module<Self>,
                res: &mut R,
                a: &A,
                ksk: &K,
                scratch: &mut ScratchArena<'s, Self>,
            )
            where
                R: poulpy_core::layouts::LWEToBackendMut<Self> + LWEInfos,
                A: poulpy_core::layouts::LWEToBackendRef<Self> + LWEInfos,
                K: GGLWEPreparedToBackendRef<Self> + GGLWEInfos,
            {
                $($helpers)*::keyswitch_lwe_keyswitch(module, res, a, ksk, scratch)
            }
        }
    };
}

impl_decryption_via_helpers!(FFT64Ref, crate::core_impl);
impl_conversion_via_helpers!(FFT64Ref, crate::core_impl);
impl_external_product_via_helpers!(FFT64Ref, crate::core_impl);
impl_keyswitching_via_helpers!(FFT64Ref, crate::core_impl);

impl_decryption_via_helpers!(NTT120Ref, crate::core_impl);
impl_conversion_via_helpers!(NTT120Ref, crate::core_impl);
impl_external_product_via_helpers!(NTT120Ref, crate::core_impl);
impl_keyswitching_via_helpers!(NTT120Ref, crate::core_impl);

#[macro_export]
macro_rules! impl_automorphism_via_defaults {
    ($be:ty) => {
        unsafe impl AutomorphismImpl<$be> for $be {
            fn glwe_automorphism_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
            where
                R: GLWEInfos,
                A: GLWEInfos,
                K: GGLWEInfos,
            {
                <Self as AutomorphismDefaults<Self>>::glwe_automorphism_tmp_bytes_default(module, res_infos, a_infos, key_infos)
            }

            fn glwe_automorphism<'s, 'r, 'a, K>(
                module: &Module<Self>,
                res: &mut poulpy_core::layouts::GLWEBackendMut<'r, Self>,
                a: &poulpy_core::layouts::GLWEBackendRef<'a, Self>,
                key: &K,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                K: poulpy_core::layouts::GetGaloisElement + GGLWEPreparedToBackendRef<Self> + GGLWEInfos,
                Self: 's,
            {
                <Self as AutomorphismDefaults<Self>>::glwe_automorphism_default(module, res, a, key, scratch)
            }

            fn glwe_automorphism_inplace<'s, 'r, K>(
                module: &Module<Self>,
                res: &mut poulpy_core::layouts::GLWEBackendMut<'r, Self>,
                key: &K,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                K: poulpy_core::layouts::GetGaloisElement + GGLWEPreparedToBackendRef<Self> + GGLWEInfos,
                Self: 's,
            {
                <Self as AutomorphismDefaults<Self>>::glwe_automorphism_inplace_default(module, res, key, scratch)
            }

            fn glwe_automorphism_add<'s, 'r, 'a, K>(
                module: &Module<Self>,
                res: &mut poulpy_core::layouts::GLWEBackendMut<'r, Self>,
                a: &poulpy_core::layouts::GLWEBackendRef<'a, Self>,
                key: &K,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                K: poulpy_core::layouts::GetGaloisElement + GGLWEPreparedToBackendRef<Self> + GGLWEInfos,
                Self: 's,
            {
                <Self as AutomorphismDefaults<Self>>::glwe_automorphism_add_default(module, res, a, key, scratch)
            }

            fn glwe_automorphism_add_inplace<'s, 'r, K>(
                module: &Module<Self>,
                res: &mut poulpy_core::layouts::GLWEBackendMut<'r, Self>,
                key: &K,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                K: poulpy_core::layouts::GetGaloisElement + GGLWEPreparedToBackendRef<Self> + GGLWEInfos,
                Self: 's,
            {
                <Self as AutomorphismDefaults<Self>>::glwe_automorphism_add_inplace_default(module, res, key, scratch)
            }

            fn glwe_automorphism_sub<'s, 'r, 'a, K>(
                module: &Module<Self>,
                res: &mut poulpy_core::layouts::GLWEBackendMut<'r, Self>,
                a: &poulpy_core::layouts::GLWEBackendRef<'a, Self>,
                key: &K,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                K: poulpy_core::layouts::GetGaloisElement + GGLWEPreparedToBackendRef<Self> + GGLWEInfos,
                Self: 's,
            {
                <Self as AutomorphismDefaults<Self>>::glwe_automorphism_sub_default(module, res, a, key, scratch)
            }

            fn glwe_automorphism_sub_negate<'s, 'r, 'a, K>(
                module: &Module<Self>,
                res: &mut poulpy_core::layouts::GLWEBackendMut<'r, Self>,
                a: &poulpy_core::layouts::GLWEBackendRef<'a, Self>,
                key: &K,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                K: poulpy_core::layouts::GetGaloisElement + GGLWEPreparedToBackendRef<Self> + GGLWEInfos,
                Self: 's,
            {
                <Self as AutomorphismDefaults<Self>>::glwe_automorphism_sub_negate_default(module, res, a, key, scratch)
            }

            fn glwe_automorphism_sub_inplace<'s, 'r, K>(
                module: &Module<Self>,
                res: &mut poulpy_core::layouts::GLWEBackendMut<'r, Self>,
                key: &K,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                K: poulpy_core::layouts::GetGaloisElement + GGLWEPreparedToBackendRef<Self> + GGLWEInfos,
                Self: 's,
            {
                <Self as AutomorphismDefaults<Self>>::glwe_automorphism_sub_inplace_default(module, res, key, scratch)
            }

            fn glwe_automorphism_sub_negate_inplace<'s, 'r, K>(
                module: &Module<Self>,
                res: &mut poulpy_core::layouts::GLWEBackendMut<'r, Self>,
                key: &K,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                K: poulpy_core::layouts::GetGaloisElement + GGLWEPreparedToBackendRef<Self> + GGLWEInfos,
                Self: 's,
            {
                <Self as AutomorphismDefaults<Self>>::glwe_automorphism_sub_negate_inplace_default(module, res, key, scratch)
            }

            fn ggsw_automorphism_tmp_bytes<R, A, K, T>(
                module: &Module<Self>,
                res_infos: &R,
                a_infos: &A,
                key_infos: &K,
                tsk_infos: &T,
            ) -> usize
            where
                R: GGSWInfos,
                A: GGSWInfos,
                K: GGLWEInfos,
                T: GGLWEInfos,
            {
                <Self as AutomorphismDefaults<Self>>::ggsw_automorphism_tmp_bytes_default(
                    module, res_infos, a_infos, key_infos, tsk_infos,
                )
            }

            fn ggsw_automorphism<'s, 'r, 'a, K, T>(
                module: &Module<Self>,
                res: &mut poulpy_core::layouts::GGSWBackendMut<'r, Self>,
                a: &poulpy_core::layouts::GGSWBackendRef<'a, Self>,
                key: &K,
                tsk: &T,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                K: poulpy_core::layouts::GetGaloisElement + GGLWEPreparedToBackendRef<Self> + GGLWEInfos,
                T: GGLWEToGGSWKeyPreparedToBackendRef<Self> + GGLWEInfos,
                Self: AutomorphismImpl<Self> + poulpy_core::oep::ConversionImpl<Self>,
            {
                <Self as AutomorphismDefaults<Self>>::ggsw_automorphism_default(module, res, a, key, tsk, scratch)
            }

            fn ggsw_automorphism_inplace<'s, 'r, K, T>(
                module: &Module<Self>,
                res: &mut poulpy_core::layouts::GGSWBackendMut<'r, Self>,
                key: &K,
                tsk: &T,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                K: poulpy_core::layouts::GetGaloisElement + GGLWEPreparedToBackendRef<Self> + GGLWEInfos,
                T: GGLWEToGGSWKeyPreparedToBackendRef<Self> + GGLWEInfos,
                Self: AutomorphismImpl<Self> + poulpy_core::oep::ConversionImpl<Self>,
            {
                <Self as AutomorphismDefaults<Self>>::ggsw_automorphism_inplace_default(module, res, key, tsk, scratch)
            }

            fn glwe_automorphism_key_automorphism_tmp_bytes<R, A, K>(
                module: &Module<Self>,
                res_infos: &R,
                a_infos: &A,
                key_infos: &K,
            ) -> usize
            where
                R: GGLWEInfos,
                A: GGLWEInfos,
                K: GGLWEInfos,
            {
                <Self as AutomorphismDefaults<Self>>::glwe_automorphism_key_automorphism_tmp_bytes_default(
                    module, res_infos, a_infos, key_infos,
                )
            }

            fn glwe_automorphism_key_automorphism<'s, R, A, K>(
                module: &Module<Self>,
                res: &mut R,
                a: &A,
                key: &K,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                R: GGLWEToBackendMut<Self> + poulpy_core::layouts::SetGaloisElement + GGLWEInfos,
                A: GGLWEToBackendRef<Self> + poulpy_core::layouts::GetGaloisElement + GGLWEInfos,
                K: GGLWEPreparedToBackendRef<Self> + poulpy_core::layouts::GetGaloisElement + GGLWEInfos,
            {
                <Self as AutomorphismDefaults<Self>>::glwe_automorphism_key_automorphism_default(module, res, a, key, scratch)
            }

            fn glwe_automorphism_key_automorphism_inplace<'s, R, K>(
                module: &Module<Self>,
                res: &mut R,
                key: &K,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                R: GGLWEToBackendMut<Self>
                    + poulpy_core::layouts::SetGaloisElement
                    + poulpy_core::layouts::GetGaloisElement
                    + GGLWEInfos,
                K: GGLWEPreparedToBackendRef<Self> + poulpy_core::layouts::GetGaloisElement + GGLWEInfos,
            {
                <Self as AutomorphismDefaults<Self>>::glwe_automorphism_key_automorphism_inplace_default(
                    module, res, key, scratch,
                )
            }
        }
    };
}

#[macro_export]
macro_rules! impl_operations_via_defaults {
    ($be:ty) => {
        unsafe impl GLWEMulConstImpl<$be> for $be {
            fn glwe_mul_const_tmp_bytes<R, A>(module: &Module<Self>, res: &R, a: &A, b_size: usize) -> usize
            where
                R: GLWEInfos,
                A: GLWEInfos,
            {
                <Self as OperationsDefaults<Self>>::glwe_mul_const_tmp_bytes_default(module, res, a, b_size)
            }

            fn glwe_mul_const<'s, R, A>(
                module: &Module<Self>,
                cnv_offset: usize,
                res: &mut poulpy_core::layouts::GLWE<R>,
                a: &poulpy_core::layouts::GLWE<A>,
                b: &[i64],
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                R: HostDataMut,
                A: HostDataRef,
                poulpy_core::layouts::GLWE<R>: poulpy_core::layouts::GLWEToBackendMut<Self>,
                poulpy_core::layouts::GLWE<A>: poulpy_core::layouts::GLWEToBackendRef<Self>,
                for<'x> <Self as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
            {
                <Self as OperationsDefaults<Self>>::glwe_mul_const_default(module, cnv_offset, res, a, b, scratch)
            }

            fn glwe_mul_const_inplace<'s, R>(
                module: &Module<Self>,
                cnv_offset: usize,
                res: &mut poulpy_core::layouts::GLWE<R>,
                b: &[i64],
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                R: HostDataMut,
                poulpy_core::layouts::GLWE<R>: poulpy_core::layouts::GLWEToBackendMut<Self>,
                for<'x> <Self as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
            {
                <Self as OperationsDefaults<Self>>::glwe_mul_const_inplace_default(module, cnv_offset, res, b, scratch)
            }
        }

        unsafe impl GLWEMulPlainImpl<$be> for $be {
            fn glwe_mul_plain_tmp_bytes<R, A, B>(module: &Module<Self>, res: &R, a: &A, b: &B) -> usize
            where
                R: GLWEInfos,
                A: GLWEInfos,
                B: GLWEInfos,
            {
                <Self as OperationsDefaults<Self>>::glwe_mul_plain_tmp_bytes_default(module, res, a, b)
            }

            fn glwe_mul_plain<'s, R, A, B>(
                module: &Module<Self>,
                cnv_offset: usize,
                res: &mut poulpy_core::layouts::GLWE<R>,
                a: &poulpy_core::layouts::GLWE<A>,
                a_effective_k: usize,
                b: &GLWEPlaintext<B>,
                b_effective_k: usize,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                R: HostDataMut,
                A: HostDataRef,
                B: HostDataRef,
                poulpy_core::layouts::GLWE<R>: poulpy_core::layouts::GLWEToBackendMut<Self>,
                poulpy_core::layouts::GLWE<A>: poulpy_core::layouts::GLWEToBackendRef<Self>,
                poulpy_core::layouts::GLWEPlaintext<B>: poulpy_core::layouts::GLWEPlaintextToBackendRef<Self>,
                for<'x> <Self as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
            {
                <Self as OperationsDefaults<Self>>::glwe_mul_plain_default(
                    module,
                    cnv_offset,
                    res,
                    a,
                    a_effective_k,
                    b,
                    b_effective_k,
                    scratch,
                )
            }

            fn glwe_mul_plain_inplace<'s, R, A>(
                module: &Module<Self>,
                cnv_offset: usize,
                res: &mut poulpy_core::layouts::GLWE<R>,
                res_effective_k: usize,
                a: &GLWEPlaintext<A>,
                a_effective_k: usize,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                R: HostDataMut,
                A: HostDataRef,
                poulpy_core::layouts::GLWE<R>: poulpy_core::layouts::GLWEToBackendMut<Self>,
                poulpy_core::layouts::GLWEPlaintext<A>: poulpy_core::layouts::GLWEPlaintextToBackendRef<Self>,
                for<'x> <Self as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
            {
                <Self as OperationsDefaults<Self>>::glwe_mul_plain_inplace_default(
                    module,
                    cnv_offset,
                    res,
                    res_effective_k,
                    a,
                    a_effective_k,
                    scratch,
                )
            }
        }

        unsafe impl GLWETensoringImpl<$be> for $be {
            fn glwe_tensor_apply_tmp_bytes<R, A, B>(module: &Module<Self>, res: &R, a: &A, b: &B) -> usize
            where
                R: GLWEInfos,
                A: GLWEInfos,
                B: GLWEInfos,
            {
                <Self as OperationsDefaults<Self>>::glwe_tensor_apply_tmp_bytes_default(module, res, a, b)
            }

            fn glwe_tensor_square_apply_tmp_bytes<R, A>(module: &Module<Self>, res: &R, a: &A) -> usize
            where
                R: GLWEInfos,
                A: GLWEInfos,
            {
                <Self as OperationsDefaults<Self>>::glwe_tensor_square_apply_tmp_bytes_default(module, res, a)
            }

            fn glwe_tensor_apply<'s, R, A, B>(
                module: &Module<Self>,
                cnv_offset: usize,
                res: &mut GLWETensor<R>,
                a: &poulpy_core::layouts::GLWE<A>,
                a_effective_k: usize,
                b: &poulpy_core::layouts::GLWE<B>,
                b_effective_k: usize,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                R: HostDataMut,
                A: HostDataRef,
                B: HostDataRef,
                GLWETensor<R>: poulpy_core::layouts::GLWEToBackendMut<Self>,
                poulpy_core::layouts::GLWE<A>: poulpy_core::layouts::GLWEToBackendRef<Self>,
                poulpy_core::layouts::GLWE<B>: poulpy_core::layouts::GLWEToBackendRef<Self>,
                for<'x> <Self as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
            {
                <Self as OperationsDefaults<Self>>::glwe_tensor_apply_default(
                    module,
                    cnv_offset,
                    res,
                    a,
                    a_effective_k,
                    b,
                    b_effective_k,
                    scratch,
                )
            }

            fn glwe_tensor_square_apply<'s, R, A>(
                module: &Module<Self>,
                cnv_offset: usize,
                res: &mut GLWETensor<R>,
                a: &poulpy_core::layouts::GLWE<A>,
                a_effective_k: usize,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                R: HostDataMut,
                A: HostDataRef,
                GLWETensor<R>: poulpy_core::layouts::GLWEToBackendMut<Self>,
                poulpy_core::layouts::GLWE<A>: poulpy_core::layouts::GLWEToBackendRef<Self>,
                for<'x> <Self as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
            {
                <Self as OperationsDefaults<Self>>::glwe_tensor_square_apply_default(
                    module,
                    cnv_offset,
                    res,
                    a,
                    a_effective_k,
                    scratch,
                )
            }

            fn glwe_tensor_relinearize<'s, R, A, B>(
                module: &Module<Self>,
                res: &mut poulpy_core::layouts::GLWE<R>,
                a: &GLWETensor<A>,
                tsk: &poulpy_core::layouts::GLWETensorKeyPrepared<B, Self>,
                tsk_size: usize,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                R: HostDataMut,
                A: HostDataRef,
                B: poulpy_hal::layouts::Data,
                poulpy_core::layouts::GLWE<R>: poulpy_core::layouts::GLWEToBackendMut<Self>,
                poulpy_core::layouts::GLWETensorKeyPrepared<B, Self>:
                    poulpy_core::layouts::GLWETensorKeyPreparedToBackendRef<Self>,
                GLWETensor<A>: poulpy_core::layouts::GLWEToBackendRef<Self>,
                for<'x> <Self as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
            {
                <Self as OperationsDefaults<Self>>::glwe_tensor_relinearize_default(module, res, a, tsk, tsk_size, scratch)
            }

            fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(module: &Module<Self>, res: &R, a: &A, tsk: &B) -> usize
            where
                R: GLWEInfos,
                A: GLWEInfos,
                B: GGLWEInfos,
            {
                <Self as OperationsDefaults<Self>>::glwe_tensor_relinearize_tmp_bytes_default(module, res, a, tsk)
            }
        }

        unsafe impl GLWERotateImpl<$be> for $be {
            fn glwe_rotate_tmp_bytes(module: &Module<Self>) -> usize {
                <Self as OperationsDefaults<Self>>::glwe_rotate_tmp_bytes_default(module)
            }
            fn glwe_rotate<'r, 'a>(
                module: &Module<Self>,
                k: i64,
                res: &mut poulpy_core::layouts::GLWEBackendMut<'r, Self>,
                a: &poulpy_core::layouts::GLWEBackendRef<'a, Self>,
            ) {
                <Self as OperationsDefaults<Self>>::glwe_rotate_default(module, k, res, a)
            }
            fn glwe_rotate_inplace<'s, 'r>(
                module: &Module<Self>,
                k: i64,
                res: &mut poulpy_core::layouts::GLWEBackendMut<'r, Self>,
                scratch: &mut ScratchArena<'s, Self>,
            ) {
                <Self as OperationsDefaults<Self>>::glwe_rotate_inplace_default(module, k, res, scratch);
            }
        }

        unsafe impl GGSWRotateImpl<$be> for $be {
            fn ggsw_rotate_tmp_bytes(module: &Module<Self>) -> usize {
                <Self as OperationsDefaults<Self>>::ggsw_rotate_tmp_bytes_default(module)
            }
            fn ggsw_rotate<'r, 'a>(
                module: &Module<Self>,
                k: i64,
                res: &mut poulpy_core::layouts::GGSWBackendMut<'r, Self>,
                a: &poulpy_core::layouts::GGSWBackendRef<'a, Self>,
            ) {
                <Self as OperationsDefaults<Self>>::ggsw_rotate_default(module, k, res, a)
            }
            fn ggsw_rotate_inplace<'s, 'r>(
                module: &Module<Self>,
                k: i64,
                res: &mut poulpy_core::layouts::GGSWBackendMut<'r, Self>,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                ScratchArena<'s, Self>: poulpy_hal::api::ScratchAvailable,
            {
                <Self as OperationsDefaults<Self>>::ggsw_rotate_inplace_default(module, k, res, scratch)
            }
        }

        unsafe impl GLWEMulXpMinusOneImpl<$be> for $be {
            fn glwe_mul_xp_minus_one<R, A>(module: &Module<Self>, k: i64, res: &mut R, a: &A)
            where
                R: GLWEToBackendMut<Self>,
                A: GLWEToBackendRef<Self>,
            {
                <Self as OperationsDefaults<Self>>::glwe_mul_xp_minus_one_default(module, k, res, a)
            }
            fn glwe_mul_xp_minus_one_inplace<'s, R>(
                module: &Module<Self>,
                k: i64,
                res: &mut R,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                R: GLWEToBackendMut<Self>,
            {
                <Self as OperationsDefaults<Self>>::glwe_mul_xp_minus_one_inplace_default(module, k, res, scratch)
            }
        }

        unsafe impl GLWEShiftImpl<$be> for $be {
            fn glwe_shift_tmp_bytes(module: &Module<Self>) -> usize {
                <Self as OperationsDefaults<Self>>::glwe_shift_tmp_bytes_default(module)
            }
            fn glwe_rsh<'s, R>(module: &Module<Self>, k: usize, res: &mut R, scratch: &mut ScratchArena<'s, Self>)
            where
                R: GLWEToBackendMut<Self>,
                ScratchArena<'s, Self>: ScratchArenaTakeCore<'s, Self>,
            {
                <Self as OperationsDefaults<Self>>::glwe_rsh_default(module, k, res, scratch)
            }
            fn glwe_lsh_inplace<'s, R>(module: &Module<Self>, res: &mut R, k: usize, scratch: &mut ScratchArena<'s, Self>)
            where
                R: GLWEToBackendMut<Self>,
            {
                <Self as OperationsDefaults<Self>>::glwe_lsh_inplace_default(module, res, k, scratch)
            }
            fn glwe_lsh<'s, R, A>(module: &Module<Self>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, Self>)
            where
                R: GLWEToBackendMut<Self>,
                A: GLWEToBackendRef<Self>,
                ScratchArena<'s, Self>: ScratchArenaTakeCore<'s, Self>,
            {
                <Self as OperationsDefaults<Self>>::glwe_lsh_default(module, res, a, k, scratch)
            }
            fn glwe_lsh_add<'s, R, A>(module: &Module<Self>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, Self>)
            where
                R: GLWEToBackendMut<Self>,
                A: GLWEToBackendRef<Self>,
                ScratchArena<'s, Self>: ScratchArenaTakeCore<'s, Self>,
            {
                <Self as OperationsDefaults<Self>>::glwe_lsh_add_default(module, res, a, k, scratch)
            }
            fn glwe_lsh_sub<'s, R, A>(module: &Module<Self>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, Self>)
            where
                R: GLWEToBackendMut<Self>,
                A: GLWEToBackendRef<Self>,
                ScratchArena<'s, Self>: ScratchArenaTakeCore<'s, Self>,
            {
                <Self as OperationsDefaults<Self>>::glwe_lsh_sub_default(module, res, a, k, scratch)
            }
        }

        unsafe impl GLWENormalizeImpl<$be> for $be {
            fn glwe_normalize_tmp_bytes(module: &Module<Self>) -> usize {
                <Self as OperationsDefaults<Self>>::glwe_normalize_tmp_bytes_default(module)
            }
            fn glwe_maybe_cross_normalize_to_ref<'a>(
                module: &Module<Self>,
                glwe: &'a poulpy_core::layouts::GLWEBackendRef<'a, Self>,
                target_base2k: usize,
                tmp_slot: &'a mut Option<poulpy_core::layouts::GLWEBackendMut<'a, Self>>,
                scratch: &'a mut ScratchArena<'a, Self>,
            ) -> poulpy_core::layouts::GLWEBackendRef<'a, Self>
            where
                ScratchArena<'a, Self>: ScratchArenaTakeCore<'a, Self>,
            {
                <Self as OperationsDefaults<Self>>::glwe_maybe_cross_normalize_to_ref_default(
                    module,
                    glwe,
                    target_base2k,
                    tmp_slot,
                    scratch,
                )
            }
            fn glwe_maybe_cross_normalize_to_mut<'a>(
                module: &Module<Self>,
                glwe: &'a mut poulpy_core::layouts::GLWEBackendMut<'a, Self>,
                target_base2k: usize,
                tmp_slot: &'a mut Option<poulpy_core::layouts::GLWEBackendMut<'a, Self>>,
                scratch: &'a mut ScratchArena<'a, Self>,
            ) -> poulpy_core::layouts::GLWEBackendMut<'a, Self>
            where
                ScratchArena<'a, Self>: ScratchArenaTakeCore<'a, Self>,
            {
                <Self as OperationsDefaults<Self>>::glwe_maybe_cross_normalize_to_mut_default(
                    module,
                    glwe,
                    target_base2k,
                    tmp_slot,
                    scratch,
                )
            }
            fn glwe_normalize<'s, 'r, 'a>(
                module: &Module<Self>,
                res: &mut poulpy_core::layouts::GLWEBackendMut<'r, Self>,
                a: &poulpy_core::layouts::GLWEBackendRef<'a, Self>,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                ScratchArena<'s, Self>: ScratchArenaTakeCore<'s, Self>,
            {
                <Self as OperationsDefaults<Self>>::glwe_normalize_default(module, res, a, scratch)
            }
            fn glwe_normalize_inplace<'s, 'r>(
                module: &Module<Self>,
                res: &mut poulpy_core::layouts::GLWEBackendMut<'r, Self>,
                scratch: &mut ScratchArena<'s, Self>,
            ) {
                <Self as OperationsDefaults<Self>>::glwe_normalize_inplace_default(module, res, scratch)
            }
        }

        unsafe impl GLWETraceImpl<$be> for $be {
            fn glwe_trace_galois_elements(module: &Module<Self>) -> Vec<i64> {
                <Self as OperationsDefaults<Self>>::glwe_trace_galois_elements_default(module)
            }
            fn glwe_trace_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
            where
                R: GLWEInfos,
                A: GLWEInfos,
                K: GGLWEInfos,
            {
                <Self as OperationsDefaults<Self>>::glwe_trace_tmp_bytes_default(module, res_infos, a_infos, key_infos)
            }
            fn glwe_trace<'s, R, A, K, H>(
                module: &Module<Self>,
                res: &mut R,
                skip: usize,
                a: &A,
                keys: &H,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                R: GLWEToBackendMut<Self> + GLWEInfos,
                A: poulpy_core::layouts::GLWEToBackendRef<Self> + GLWEInfos,
                K: GGLWEPreparedToBackendRef<Self> + poulpy_core::layouts::GetGaloisElement + GGLWEInfos,
                H: poulpy_core::layouts::GLWEAutomorphismKeyHelper<K, Self>,
                Self: 's,
                for<'x> ScratchArena<'x, Self>: ScratchArenaTakeCore<'x, Self>,
                for<'x> Self: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
            {
                <Self as OperationsDefaults<Self>>::glwe_trace_default(module, res, skip, a, keys, scratch)
            }
            fn glwe_trace_inplace<'s, R, K, H>(
                module: &Module<Self>,
                res: &mut R,
                skip: usize,
                keys: &H,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                R: poulpy_core::layouts::GLWEToBackendMut<Self> + GLWEInfos,
                K: GGLWEPreparedToBackendRef<Self> + poulpy_core::layouts::GetGaloisElement + GGLWEInfos,
                H: poulpy_core::layouts::GLWEAutomorphismKeyHelper<K, Self>,
                Self: 's,
                for<'x> ScratchArena<'x, Self>: ScratchArenaTakeCore<'x, Self>,
                for<'x> Self: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
            {
                <Self as OperationsDefaults<Self>>::glwe_trace_inplace_default(module, res, skip, keys, scratch)
            }
        }

        unsafe impl GLWEPackImpl<$be> for $be {
            fn glwe_pack_galois_elements(module: &Module<Self>) -> Vec<i64> {
                <Self as OperationsDefaults<Self>>::glwe_pack_galois_elements_default(module)
            }
            fn glwe_pack_tmp_bytes<R, K>(module: &Module<Self>, res: &R, key: &K) -> usize
            where
                R: GLWEInfos,
                K: GGLWEInfos,
            {
                <Self as OperationsDefaults<Self>>::glwe_pack_tmp_bytes_default(module, res, key)
            }
            fn glwe_pack<'s, R, A, K, H>(
                module: &Module<Self>,
                res: &mut R,
                a: std::collections::HashMap<usize, &mut A>,
                log_gap_out: usize,
                keys: &H,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                R: GLWEToBackendMut<Self> + GLWEInfos,
                A: poulpy_core::layouts::GLWEToBackendMut<Self> + GLWEInfos,
                K: GGLWEPreparedToBackendRef<Self> + poulpy_core::layouts::GetGaloisElement + GGLWEInfos,
                H: poulpy_core::layouts::GLWEAutomorphismKeyHelper<K, Self>,
                Self: 's,
                for<'x> ScratchArena<'x, Self>: ScratchArenaTakeCore<'x, Self>,
                for<'x> Self: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
            {
                <Self as OperationsDefaults<Self>>::glwe_pack_default(module, res, a, log_gap_out, keys, scratch)
            }
            fn packer_add<'s, A, K, H>(
                module: &Module<Self>,
                packer: &mut poulpy_core::GLWEPacker,
                a: Option<&A>,
                i: usize,
                auto_keys: &H,
                scratch: &mut ScratchArena<'s, Self>,
            ) where
                A: poulpy_core::layouts::GLWEToBackendRef<Self> + GLWEInfos,
                K: GGLWEPreparedToBackendRef<Self> + poulpy_core::layouts::GetGaloisElement + GGLWEInfos,
                H: poulpy_core::layouts::GLWEAutomorphismKeyHelper<K, Self>,
                poulpy_core::layouts::GLWE<Vec<u8>>:
                    poulpy_core::layouts::GLWEToBackendMut<Self> + poulpy_core::layouts::GLWEToBackendRef<Self>,
                ScratchArena<'s, Self>: ScratchArenaTakeCore<'s, Self>,
            {
                <Self as OperationsDefaults<Self>>::packer_add_default(module, packer, a, i, auto_keys, scratch)
            }
        }
    };
}

impl_automorphism_via_defaults!(FFT64Ref);
impl_operations_via_defaults!(FFT64Ref);

impl_automorphism_via_defaults!(NTT120Ref);
impl_operations_via_defaults!(NTT120Ref);
