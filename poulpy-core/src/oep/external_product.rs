use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    ScratchArenaTakeCore,
    external_product::{GGLWEExternalProductDefault, GGSWExternalProductDefault, GLWEExternalProductDefault},
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGLWEToMut, GGLWEToRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef,
        GGSWToMut, GGSWToRef, GLWEBackendMut, GLWEBackendRef, GLWEInfos, prepared::GGSWPreparedToBackendRef,
    },
};

/// Backend hook for GLWE external products.
///
/// # Safety
/// Implementors must preserve the semantics, scratch requirements, and aliasing
/// guarantees expected by the public and default external-product layers.
pub unsafe trait GLWEExternalProductImpl<BE: Backend>: Backend {
    fn glwe_external_product_tmp_bytes<R, A, G>(module: &Module<BE>, res_infos: &R, a_infos: &A, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        G: GGSWInfos;

    fn glwe_external_product<'s, 'r, 'a, G>(
        module: &Module<BE>,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &GLWEBackendRef<'a, BE>,
        ggsw: &G,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        G: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;

    fn glwe_external_product_inplace<'s, 'r, G>(
        module: &Module<BE>,
        res: &mut GLWEBackendMut<'r, BE>,
        ggsw: &G,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        G: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;
}

/// Backend hook for batched GGLWE external products.
///
/// # Safety
/// Implementors must preserve the semantics, scratch requirements, and aliasing
/// guarantees expected by the public and default external-product layers.
pub unsafe trait GGLWEExternalProductImpl<BE: Backend>: Backend {
    fn gglwe_external_product_tmp_bytes<R, A, B>(module: &Module<BE>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos;

    fn gglwe_external_product<'s, R, A, B>(module: &Module<BE>, res: &mut R, a: &A, b: &B, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToMut + GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToRef + GGLWEToBackendRef<BE> + GGLWEInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;

    fn gglwe_external_product_inplace<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToMut + GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;
}

/// Backend hook for GGSW external products.
///
/// # Safety
/// Implementors must preserve the semantics, scratch requirements, and aliasing
/// guarantees expected by the public and default external-product layers.
pub unsafe trait GGSWExternalProductImpl<BE: Backend>: Backend {
    fn ggsw_external_product_tmp_bytes<R, A, B>(module: &Module<BE>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos;

    fn ggsw_external_product<'s, R, A, B>(module: &Module<BE>, res: &mut R, a: &A, b: &B, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToMut + GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToRef + GGSWToBackendRef<BE> + GGSWInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;

    fn ggsw_external_product_inplace<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToMut + GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;
}

#[doc(hidden)]
pub trait GLWEExternalProductDefaults<BE: Backend>: Backend {
    fn glwe_external_product_tmp_bytes<R, A, G>(module: &Module<BE>, res_infos: &R, a_infos: &A, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        G: GGSWInfos;

    fn glwe_external_product<'s, 'r, 'a, G>(
        module: &Module<BE>,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &GLWEBackendRef<'a, BE>,
        ggsw: &G,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        G: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;

    fn glwe_external_product_inplace<'s, 'r, G>(
        module: &Module<BE>,
        res: &mut GLWEBackendMut<'r, BE>,
        ggsw: &G,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        G: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;
}

#[doc(hidden)]
pub trait GGLWEExternalProductDefaults<BE: Backend>: Backend {
    fn gglwe_external_product_tmp_bytes<R, A, B>(module: &Module<BE>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos;

    fn gglwe_external_product<'s, R, A, B>(module: &Module<BE>, res: &mut R, a: &A, b: &B, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToMut + GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToRef + GGLWEToBackendRef<BE> + GGLWEInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;

    fn gglwe_external_product_inplace<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToMut + GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;
}

#[doc(hidden)]
pub trait GGSWExternalProductDefaults<BE: Backend>: Backend {
    fn ggsw_external_product_tmp_bytes<R, A, B>(module: &Module<BE>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos;

    fn ggsw_external_product<'s, R, A, B>(module: &Module<BE>, res: &mut R, a: &A, b: &B, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToMut + GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToRef + GGSWToBackendRef<BE> + GGSWInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;

    fn ggsw_external_product_inplace<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToMut + GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;
}

impl<BE: Backend> GLWEExternalProductDefaults<BE> for BE
where
    Module<BE>: GLWEExternalProductDefault<BE>,
{
    fn glwe_external_product_tmp_bytes<R, A, G>(module: &Module<BE>, res_infos: &R, a_infos: &A, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        G: GGSWInfos,
    {
        <Module<BE> as GLWEExternalProductDefault<BE>>::glwe_external_product_tmp_bytes_default(
            module, res_infos, a_infos, ggsw_infos,
        )
    }

    fn glwe_external_product<'s, 'r, 'a, G>(
        module: &Module<BE>,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &GLWEBackendRef<'a, BE>,
        ggsw: &G,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        G: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        <Module<BE> as GLWEExternalProductDefault<BE>>::glwe_external_product_default(module, res, a, ggsw, scratch)
    }

    fn glwe_external_product_inplace<'s, 'r, G>(
        module: &Module<BE>,
        res: &mut GLWEBackendMut<'r, BE>,
        ggsw: &G,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        G: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        <Module<BE> as GLWEExternalProductDefault<BE>>::glwe_external_product_assign_default(module, res, ggsw, scratch)
    }
}

impl<BE: Backend> GGLWEExternalProductDefaults<BE> for BE
where
    Module<BE>: GGLWEExternalProductDefault<BE>,
{
    fn gglwe_external_product_tmp_bytes<R, A, B>(module: &Module<BE>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos,
    {
        <Module<BE> as GGLWEExternalProductDefault<BE>>::gglwe_external_product_tmp_bytes_default(
            module, res_infos, a_infos, b_infos,
        )
    }

    fn gglwe_external_product<'s, R, A, B>(module: &Module<BE>, res: &mut R, a: &A, b: &B, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToMut + GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToRef + GGLWEToBackendRef<BE> + GGLWEInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        <Module<BE> as GGLWEExternalProductDefault<BE>>::gglwe_external_product_default(module, res, a, b, scratch)
    }

    fn gglwe_external_product_inplace<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToMut + GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        <Module<BE> as GGLWEExternalProductDefault<BE>>::gglwe_external_product_assign_default(module, res, a, scratch)
    }
}

impl<BE: Backend> GGSWExternalProductDefaults<BE> for BE
where
    Module<BE>: GGSWExternalProductDefault<BE>,
{
    fn ggsw_external_product_tmp_bytes<R, A, B>(module: &Module<BE>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos,
    {
        <Module<BE> as GGSWExternalProductDefault<BE>>::ggsw_external_product_tmp_bytes_default(
            module, res_infos, a_infos, b_infos,
        )
    }

    fn ggsw_external_product<'s, R, A, B>(module: &Module<BE>, res: &mut R, a: &A, b: &B, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToMut + GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToRef + GGSWToBackendRef<BE> + GGSWInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        <Module<BE> as GGSWExternalProductDefault<BE>>::ggsw_external_product_default(module, res, a, b, scratch)
    }

    fn ggsw_external_product_inplace<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToMut + GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        <Module<BE> as GGSWExternalProductDefault<BE>>::ggsw_external_product_assign_default(module, res, a, scratch)
    }
}
