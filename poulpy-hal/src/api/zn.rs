use crate::{
    layouts::{Backend, Scratch, ZnToMut},
    reference::zn::zn_normalize_tmp_bytes,
    source::Source,
};

pub trait ZnNormalizeTmpBytes {
    fn zn_normalize_tmp_bytes(&self, n: usize) -> usize {
        zn_normalize_tmp_bytes(n)
    }
}

pub trait ZnNormalizeInplace<B: Backend> {
    /// Normalizes the selected column of `a`.
    fn zn_normalize_inplace<R>(&self, n: usize, base2k: usize, res: &mut R, res_col: usize, scratch: &mut Scratch<B>)
    where
        R: ZnToMut;
}

pub trait ZnFillUniform {
    /// Fills the first `size` size with uniform values in \[-2^{base2k-1}, 2^{base2k-1}\]
    fn zn_fill_uniform<R>(&self, n: usize, base2k: usize, res: &mut R, res_col: usize, source: &mut Source)
    where
        R: ZnToMut;
}

#[allow(clippy::too_many_arguments)]
pub trait ZnFillNormal {
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
        R: ZnToMut;
}

#[allow(clippy::too_many_arguments)]
pub trait ZnAddNormal {
    /// Adds a discrete normal vector scaled by 2^{-k} with the provided standard deviation and bounded to \[-bound, bound\].
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
        R: ZnToMut;
}
