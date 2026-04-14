use poulpy_hal::layouts::{Backend, Module, Scratch};

use crate::{
    ScratchTakeCore,
    conversion::{GGSWExpandRowsDefault, GGSWFromGGLWEDefault, GLWEFromLWEDefault, LWEFromGLWEDefault},
    layouts::{
        GGLWEInfos, GGLWEPreparedToRef, GGLWEToGGSWKeyPreparedToRef, GGLWEToRef, GGSWInfos, GGSWToMut, GLWEInfos, GLWEToMut,
        GLWEToRef, LWEInfos, LWEToMut, LWEToRef,
    },
};

#[doc(hidden)]
pub trait CoreConversionDefaults<BE: Backend>: Backend {
    fn glwe_from_lwe_tmp_bytes_default<R, A, K>(module: &Module<BE>, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn glwe_from_lwe_default<R, A, K>(module: &Module<BE>, res: &mut R, lwe: &A, ksk: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: LWEToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn lwe_from_glwe_tmp_bytes_default<R, A, K>(module: &Module<BE>, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn lwe_from_glwe_default<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, a_idx: usize, key: &K, scratch: &mut Scratch<BE>)
    where
        R: LWEToMut,
        A: GLWEToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ggsw_from_gglwe_tmp_bytes_default<R, A>(module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_from_gglwe_default<R, A, T>(module: &Module<BE>, res: &mut R, a: &A, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGLWEToRef,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ggsw_expand_rows_tmp_bytes_default<R, A>(module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_expand_row_default<R, T>(module: &Module<BE>, res: &mut R, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;
}

impl<BE: Backend> CoreConversionDefaults<BE> for BE
where
    Module<BE>: GLWEFromLWEDefault<BE> + LWEFromGLWEDefault<BE> + GGSWFromGGLWEDefault<BE> + GGSWExpandRowsDefault<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_from_lwe_tmp_bytes_default<R, A, K>(module: &Module<BE>, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        <Module<BE> as GLWEFromLWEDefault<BE>>::glwe_from_lwe_tmp_bytes_default(module, glwe_infos, lwe_infos, key_infos)
    }

    fn glwe_from_lwe_default<R, A, K>(module: &Module<BE>, res: &mut R, lwe: &A, ksk: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: LWEToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        <Module<BE> as GLWEFromLWEDefault<BE>>::glwe_from_lwe_default(module, res, lwe, ksk, scratch)
    }

    fn lwe_from_glwe_tmp_bytes_default<R, A, K>(module: &Module<BE>, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        <Module<BE> as LWEFromGLWEDefault<BE>>::lwe_from_glwe_tmp_bytes_default(module, lwe_infos, glwe_infos, key_infos)
    }

    fn lwe_from_glwe_default<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, a_idx: usize, key: &K, scratch: &mut Scratch<BE>)
    where
        R: LWEToMut,
        A: GLWEToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as LWEFromGLWEDefault<BE>>::lwe_from_glwe_default(module, res, a, a_idx, key, scratch)
    }

    fn ggsw_from_gglwe_tmp_bytes_default<R, A>(module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos,
    {
        <Module<BE> as GGSWFromGGLWEDefault<BE>>::ggsw_from_gglwe_tmp_bytes_default(module, res_infos, tsk_infos)
    }

    fn ggsw_from_gglwe_default<R, A, T>(module: &Module<BE>, res: &mut R, a: &A, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGLWEToRef,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GGSWFromGGLWEDefault<BE>>::ggsw_from_gglwe_default(module, res, a, tsk, scratch)
    }

    fn ggsw_expand_rows_tmp_bytes_default<R, A>(module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos,
    {
        <Module<BE> as GGSWExpandRowsDefault<BE>>::ggsw_expand_rows_tmp_bytes_default(module, res_infos, tsk_infos)
    }

    fn ggsw_expand_row_default<R, T>(module: &Module<BE>, res: &mut R, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GGSWExpandRowsDefault<BE>>::ggsw_expand_row_default(module, res, tsk, scratch)
    }
}

#[macro_export]
macro_rules! impl_core_conversion_default_methods {
    ($be:ty) => {
        fn glwe_from_lwe_tmp_bytes<R, A, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            glwe_infos: &R,
            lwe_infos: &A,
            key_infos: &K,
        ) -> usize
        where
            R: $crate::layouts::GLWEInfos,
            A: $crate::layouts::LWEInfos,
            K: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreConversionDefaults<$be>>::glwe_from_lwe_tmp_bytes_default(
                module, glwe_infos, lwe_infos, key_infos,
            )
        }

        fn glwe_from_lwe<R, A, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            lwe: &A,
            ksk: &K,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut,
            A: $crate::layouts::LWEToRef,
            K: $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreConversionDefaults<$be>>::glwe_from_lwe_default(module, res, lwe, ksk, scratch)
        }

        fn lwe_from_glwe_tmp_bytes<R, A, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            lwe_infos: &R,
            glwe_infos: &A,
            key_infos: &K,
        ) -> usize
        where
            R: $crate::layouts::LWEInfos,
            A: $crate::layouts::GLWEInfos,
            K: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreConversionDefaults<$be>>::lwe_from_glwe_tmp_bytes_default(
                module, lwe_infos, glwe_infos, key_infos,
            )
        }

        fn lwe_from_glwe<R, A, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: &A,
            a_idx: usize,
            key: &K,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::LWEToMut,
            A: $crate::layouts::GLWEToRef,
            K: $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreConversionDefaults<$be>>::lwe_from_glwe_default(module, res, a, a_idx, key, scratch)
        }

        fn ggsw_from_gglwe_tmp_bytes<R, A>(module: &poulpy_hal::layouts::Module<$be>, res_infos: &R, tsk_infos: &A) -> usize
        where
            R: $crate::layouts::GGSWInfos,
            A: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreConversionDefaults<$be>>::ggsw_from_gglwe_tmp_bytes_default(module, res_infos, tsk_infos)
        }

        fn ggsw_from_gglwe<R, A, T>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: &A,
            tsk: &T,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGSWToMut,
            A: $crate::layouts::GGLWEToRef,
            T: $crate::layouts::GGLWEToGGSWKeyPreparedToRef<$be>,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreConversionDefaults<$be>>::ggsw_from_gglwe_default(module, res, a, tsk, scratch)
        }

        fn ggsw_expand_rows_tmp_bytes<R, A>(module: &poulpy_hal::layouts::Module<$be>, res_infos: &R, tsk_infos: &A) -> usize
        where
            R: $crate::layouts::GGSWInfos,
            A: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreConversionDefaults<$be>>::ggsw_expand_rows_tmp_bytes_default(module, res_infos, tsk_infos)
        }

        fn ggsw_expand_row<R, T>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            tsk: &T,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGSWToMut,
            T: $crate::layouts::GGLWEToGGSWKeyPreparedToRef<$be>,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreConversionDefaults<$be>>::ggsw_expand_row_default(module, res, tsk, scratch)
        }
    };
}
