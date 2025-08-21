use rand_distr::Distribution;

use crate::{
    layouts::{Backend, Scratch, ZnToMut},
    source::Source,
};

pub trait ZnNormalizeInplace<B: Backend> {
    /// Normalizes the selected column of `a`.
    fn zn_normalize_inplace<A>(&self, n: usize, basek: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: ZnToMut;
}

pub trait ZnFillUniform {
    /// Fills the first `size` size with uniform values in \[-2^{basek-1}, 2^{basek-1}\]
    fn zn_fill_uniform<R>(&self, n: usize, basek: usize, res: &mut R, res_col: usize, k: usize, source: &mut Source)
    where
        R: ZnToMut;
}

#[allow(clippy::too_many_arguments)]
pub trait ZnFillDistF64 {
    fn zn_fill_dist_f64<R, D: Distribution<f64>>(
        &self,
        n: usize,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) where
        R: ZnToMut;
}

#[allow(clippy::too_many_arguments)]
pub trait ZnAddDistF64 {
    /// Adds vector sampled according to the provided distribution, scaled by 2^{-k} and bounded to \[-bound, bound\].
    fn zn_add_dist_f64<R, D: Distribution<f64>>(
        &self,
        n: usize,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) where
        R: ZnToMut;
}

#[allow(clippy::too_many_arguments)]
pub trait ZnFillNormal {
    fn zn_fill_normal<R>(
        &self,
        n: usize,
        basek: usize,
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
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: ZnToMut;
}
