use poulpy_hal::layouts::{Backend, DataMut, Module, Scratch, ZnxZero};

use crate::{
    GLWEExternalProduct, ScratchTakeCore,
    layouts::{
        GGLWE, GGLWEInfos, GGLWEToMut, GGLWEToRef, GGSWInfos, GLWEInfos, GLWESwitchingKey,
        prepared::{GGSWPrepared, GGSWPreparedToRef},
    },
};

pub trait GGLWEExternalProduct<BE: Backend>
where
    Self: GLWEExternalProduct<BE>,
{
    fn gglwe_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos,
    {
        self.glwe_external_product_tmp_bytes(res_infos, a_infos, b_infos)
    }

    fn gglwe_external_product<R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut,
        A: GGLWEToRef,
        B: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GGLWE<&[u8]> = &a.to_ref();
        let b: &GGSWPrepared<&[u8], BE> = &b.to_ref();

        assert_eq!(
            res.rank_in(),
            a.rank_in(),
            "res input rank_in: {} != a input rank_in: {}",
            res.rank_in(),
            a.rank_in()
        );
        assert_eq!(
            a.rank_out(),
            b.rank(),
            "a output rank_out: {} != b rank: {}",
            a.rank_out(),
            b.rank()
        );
        assert_eq!(
            res.rank_out(),
            b.rank(),
            "res output rank_out: {} != b rank: {}",
            res.rank_out(),
            b.rank()
        );

        for row in 0..res.dnum().into() {
            for col in 0..res.rank_in().into() {
                self.glwe_external_product(&mut res.at_mut(row, col), &a.at(row, col), b, scratch);
            }
        }

        for row in res.dnum().min(a.dnum()).into()..res.dnum().into() {
            for col in 0..res.rank_in().into() {
                res.at_mut(row, col).data_mut().zero();
            }
        }
    }

    fn gglwe_external_product_inplace<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut,
        A: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GGSWPrepared<&[u8], BE> = &a.to_ref();

        assert_eq!(
            res.rank_out(),
            a.rank(),
            "res output rank: {} != a rank: {}",
            res.rank_out(),
            a.rank()
        );

        for row in 0..res.dnum().into() {
            for col in 0..res.rank_in().into() {
                self.glwe_external_product_inplace(&mut res.at_mut(row, col), a, scratch);
            }
        }
    }
}

impl<BE: Backend> GGLWEExternalProduct<BE> for Module<BE> where Self: GLWEExternalProduct<BE> {}

impl GLWESwitchingKey<Vec<u8>> {
    pub fn external_product_tmp_bytes<R, A, B, M, BE: Backend>(module: &M, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos,
        M: GGLWEExternalProduct<BE>,
    {
        module.gglwe_external_product_tmp_bytes(res_infos, a_infos, b_infos)
    }
}

impl<DataSelf: DataMut> GLWESwitchingKey<DataSelf> {
    pub fn external_product<A, B, M, BE: Backend>(&mut self, module: &M, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        M: GGLWEExternalProduct<BE>,
        A: GGLWEToRef,
        B: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.gglwe_external_product(self, a, b, scratch);
    }

    pub fn external_product_inplace<A, M, BE: Backend>(&mut self, module: &M, a: &A, scratch: &mut Scratch<BE>)
    where
        M: GGLWEExternalProduct<BE>,
        A: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.gglwe_external_product_inplace(self, a, scratch);
    }
}
