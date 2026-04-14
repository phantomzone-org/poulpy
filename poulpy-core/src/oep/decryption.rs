use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

use crate::{
    ScratchTakeCore,
    decryption::{GLWEDecryptDefault, GLWETensorDecryptDefault, LWEDecryptDefault},
    layouts::{
        GLWEInfos, GLWEPlaintext, GLWEPlaintextToMut, GLWESecretPrepared, GLWESecretPreparedToRef, GLWESecretTensorPrepared,
        GLWETensor, LWEInfos, LWEPlaintextToMut, LWESecretToRef, LWEToRef, SetGLWEInfos, SetLWEInfos,
    },
};

#[doc(hidden)]
pub trait CoreDecryptionDefaults<BE: Backend>: Backend {
    fn glwe_decrypt_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_decrypt_default<R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        R: crate::layouts::GLWEToRef + GLWEInfos,
        P: GLWEPlaintextToMut + GLWEInfos + SetGLWEInfos,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos;

    fn lwe_decrypt_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_decrypt_default<R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        R: LWEToRef,
        P: LWEPlaintextToMut + SetLWEInfos + LWEInfos,
        S: LWESecretToRef,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_tensor_decrypt_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_tensor_decrypt_default<R, P, PM, S0, S1>(
        module: &Module<BE>,
        res: &GLWETensor<R>,
        pt: &mut GLWEPlaintext<P, PM>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataRef,
        P: DataMut,
        S0: DataRef,
        S1: DataRef;
}

impl<BE: Backend> CoreDecryptionDefaults<BE> for BE
where
    Module<BE>: GLWEDecryptDefault<BE> + LWEDecryptDefault<BE> + GLWETensorDecryptDefault<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_decrypt_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        <Module<BE> as GLWEDecryptDefault<BE>>::glwe_decrypt_tmp_bytes_default(module, infos)
    }

    fn glwe_decrypt_default<R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        R: crate::layouts::GLWEToRef + GLWEInfos,
        P: GLWEPlaintextToMut + GLWEInfos + SetGLWEInfos,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
    {
        <Module<BE> as GLWEDecryptDefault<BE>>::glwe_decrypt_default(module, res, pt, sk, scratch)
    }

    fn lwe_decrypt_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: LWEInfos,
    {
        <Module<BE> as LWEDecryptDefault<BE>>::lwe_decrypt_tmp_bytes_default(module, infos)
    }

    fn lwe_decrypt_default<R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        R: LWEToRef,
        P: LWEPlaintextToMut + SetLWEInfos + LWEInfos,
        S: LWESecretToRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as LWEDecryptDefault<BE>>::lwe_decrypt_default(module, res, pt, sk, scratch)
    }

    fn glwe_tensor_decrypt_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        <Module<BE> as GLWETensorDecryptDefault<BE>>::glwe_tensor_decrypt_tmp_bytes_default(module, infos)
    }

    fn glwe_tensor_decrypt_default<R, P, PM, S0, S1>(
        module: &Module<BE>,
        res: &GLWETensor<R>,
        pt: &mut GLWEPlaintext<P, PM>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataRef,
        P: DataMut,
        S0: DataRef,
        S1: DataRef,
    {
        <Module<BE> as GLWETensorDecryptDefault<BE>>::glwe_tensor_decrypt_default(module, res, pt, sk, sk_tensor, scratch)
    }
}

#[macro_export]
macro_rules! impl_core_decryption_default_methods {
    ($be:ty) => {
        fn glwe_decrypt_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GLWEInfos,
        {
            <$be as $crate::oep::CoreDecryptionDefaults<$be>>::glwe_decrypt_tmp_bytes_default(module, infos)
        }

        fn glwe_decrypt<R, P, S>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &R,
            pt: &mut P,
            sk: &S,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToRef + $crate::layouts::GLWEInfos,
            P: $crate::layouts::GLWEPlaintextToMut + $crate::layouts::GLWEInfos + $crate::layouts::SetGLWEInfos,
            S: $crate::layouts::GLWESecretPreparedToRef<$be> + $crate::layouts::GLWEInfos,
        {
            <$be as $crate::oep::CoreDecryptionDefaults<$be>>::glwe_decrypt_default(module, res, pt, sk, scratch)
        }

        fn lwe_decrypt_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::LWEInfos,
        {
            <$be as $crate::oep::CoreDecryptionDefaults<$be>>::lwe_decrypt_tmp_bytes_default(module, infos)
        }

        fn lwe_decrypt<R, P, S>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &R,
            pt: &mut P,
            sk: &S,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::LWEToRef,
            P: $crate::layouts::LWEPlaintextToMut + $crate::layouts::SetLWEInfos + $crate::layouts::LWEInfos,
            S: $crate::layouts::LWESecretToRef,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreDecryptionDefaults<$be>>::lwe_decrypt_default(module, res, pt, sk, scratch)
        }

        fn glwe_tensor_decrypt_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GLWEInfos,
        {
            <$be as $crate::oep::CoreDecryptionDefaults<$be>>::glwe_tensor_decrypt_tmp_bytes_default(module, infos)
        }

        fn glwe_tensor_decrypt<R, P, PM, S0, S1>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &$crate::layouts::GLWETensor<R>,
            pt: &mut $crate::layouts::GLWEPlaintext<P, PM>,
            sk: &$crate::layouts::GLWESecretPrepared<S0, $be>,
            sk_tensor: &$crate::layouts::GLWESecretTensorPrepared<S1, $be>,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: poulpy_hal::layouts::DataRef,
            P: poulpy_hal::layouts::DataMut,
            S0: poulpy_hal::layouts::DataRef,
            S1: poulpy_hal::layouts::DataRef,
        {
            <$be as $crate::oep::CoreDecryptionDefaults<$be>>::glwe_tensor_decrypt_default(
                module, res, pt, sk, sk_tensor, scratch,
            )
        }
    };
}
