use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, DataMut, Module, Scratch, ZnxZero},
};

use crate::{
    GLWEExternalProduct, ScratchTakeCore,
    layouts::{
        GGSW, GGSWInfos, GGSWToMut, GGSWToRef, GLWEInfos, LWEInfos,
        prepared::{GGSWPrepared, GGSWPreparedToRef},
    },
};

pub trait GGSWExternalProduct<BE: Backend>
where
    Self: GLWEExternalProduct<BE>,
{
    fn ggsw_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos,
    {
        self.glwe_external_product_tmp_bytes(res_infos, a_infos, b_infos)
    }

    fn ggsw_external_product<R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGSWToRef,
        B: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let a: &GGSW<&[u8]> = &a.to_ref();
        let b: &GGSWPrepared<&[u8], BE> = &b.to_ref();

        assert_eq!(
            res.rank(),
            a.rank(),
            "res rank: {} != a rank: {}",
            res.rank(),
            a.rank()
        );
        assert_eq!(
            res.rank(),
            b.rank(),
            "res rank: {} != b rank: {}",
            res.rank(),
            b.rank()
        );

        assert!(scratch.available() >= self.ggsw_external_product_tmp_bytes(res, a, b));

        let min_dnum: usize = res.dnum().min(a.dnum()).into();

        for row in 0..min_dnum {
            for col in 0..(res.rank() + 1).into() {
                self.glwe_external_product(&mut res.at_mut(row, col), &a.at(row, col), b, scratch);
            }
        }

        for row in min_dnum..res.dnum().into() {
            for col in 0..(res.rank() + 1).into() {
                res.at_mut(row, col).data.zero();
            }
        }
    }

    fn ggsw_external_product_inplace<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let a: &GGSWPrepared<&[u8], BE> = &a.to_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(
            res.rank(),
            a.rank(),
            "res rank: {} != a rank: {}",
            res.rank(),
            a.rank()
        );

        for row in 0..res.dnum().into() {
            for col in 0..(res.rank() + 1).into() {
                self.glwe_external_product_inplace(&mut res.at_mut(row, col), a, scratch);
            }
        }
    }
}

impl<BE: Backend> GGSWExternalProduct<BE> for Module<BE> where Self: GLWEExternalProduct<BE> {}

impl GGSW<Vec<u8>> {
    pub fn external_product_tmp_bytes<R, A, B, M, BE: Backend>(
        &self,
        module: &M,
        res_infos: &R,
        a_infos: &A,
        b_infos: &B,
    ) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos,
        M: GGSWExternalProduct<BE>,
    {
        module.ggsw_external_product_tmp_bytes(res_infos, a_infos, b_infos)
    }
}

impl<DataSelf: DataMut> GGSW<DataSelf> {
    pub fn external_product<A, B, M, BE: Backend>(&mut self, module: &M, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        M: GGSWExternalProduct<BE>,
        A: GGSWToRef,
        B: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.ggsw_external_product(self, a, b, scratch);
    }

    pub fn external_product_inplace<A, M, BE: Backend>(&mut self, module: &M, a: &A, scratch: &mut Scratch<BE>)
    where
        M: GGSWExternalProduct<BE>,
        A: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.ggsw_external_product_inplace(self, a, scratch);
    }
}
