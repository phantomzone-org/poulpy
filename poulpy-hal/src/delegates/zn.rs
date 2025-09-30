use crate::{
    api::{ZnAddNormal, ZnFillNormal, ZnFillUniform, ZnNormalizeInplace, ZnNormalizeTmpBytes},
    layouts::{Backend, Module, Scratch, ZnToMut},
    oep::{ZnAddNormalImpl, ZnFillNormalImpl, ZnFillUniformImpl, ZnNormalizeInplaceImpl, ZnNormalizeTmpBytesImpl},
    source::Source,
};

impl<B> ZnNormalizeTmpBytes for Module<B>
where
    B: Backend + ZnNormalizeTmpBytesImpl<B>,
{
    fn zn_normalize_tmp_bytes(&self, n: usize) -> usize {
        B::zn_normalize_tmp_bytes_impl(n)
    }
}

impl<B> ZnNormalizeInplace<B> for Module<B>
where
    B: Backend + ZnNormalizeInplaceImpl<B>,
{
    fn zn_normalize_inplace<A>(&self, n: usize, base2k: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: ZnToMut,
    {
        B::zn_normalize_inplace_impl(n, base2k, a, a_col, scratch)
    }
}

impl<B> ZnFillUniform for Module<B>
where
    B: Backend + ZnFillUniformImpl<B>,
{
    fn zn_fill_uniform<R>(&self, n: usize, base2k: usize, res: &mut R, res_col: usize, source: &mut Source)
    where
        R: ZnToMut,
    {
        B::zn_fill_uniform_impl(n, base2k, res, res_col, source);
    }
}

impl<B> ZnFillNormal for Module<B>
where
    B: Backend + ZnFillNormalImpl<B>,
{
    fn zn_fill_normal<R>(
        &self,
        n: usize,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: ZnToMut,
    {
        B::zn_fill_normal_impl(n, base2k, res, res_col, k, source, sigma, bound);
    }
}

impl<B> ZnAddNormal for Module<B>
where
    B: Backend + ZnAddNormalImpl<B>,
{
    fn zn_add_normal<R>(
        &self,
        n: usize,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: ZnToMut,
    {
        B::zn_add_normal_impl(n, base2k, res, res_col, k, source, sigma, bound);
    }
}
