use poulpy_hal::{
    api::VecZnxZeroBackend,
    layouts::{Backend, Module, ScratchArena, ZnxInfos},
};

pub use crate::api::GGLWEExternalProduct;
use crate::{
    GLWEExternalProduct, ScratchArenaTakeCore,
    layouts::{GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWInfos, prepared::GGSWPreparedToBackendRef},
};

#[doc(hidden)]
pub trait GGLWEExternalProductDefault<BE: Backend>
where
    Self: GLWEExternalProduct<BE> + VecZnxZeroBackend<BE>,
{
    fn gglwe_external_product_tmp_bytes_default<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos,
    {
        self.glwe_external_product_tmp_bytes(res_infos, a_infos, b_infos)
    }

    fn gglwe_external_product_default<'s, R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
        BE: 's,
    {
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
        assert_eq!(res.base2k(), a.base2k());
        assert!(
            scratch.available() >= self.gglwe_external_product_tmp_bytes_default(res, a, b),
            "scratch.available(): {} < GGLWEExternalProduct::gglwe_external_product_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_external_product_tmp_bytes_default(res, a, b)
        );

        let min_dnum: usize = res.dnum().min(a.dnum()).into();
        let res_dnum: usize = res.dnum().into();
        let res_rank_in: usize = res.rank_in().into();
        {
            let mut res = res.to_backend_mut();
            let a = a.to_backend_ref();
            for row in 0..min_dnum {
                for col in 0..res_rank_in {
                    self.glwe_external_product(
                        &mut crate::layouts::gglwe_at_backend_mut_from_mut::<BE>(&mut res, row, col),
                        &crate::layouts::gglwe_at_backend_ref_from_ref::<BE>(&a, row, col),
                        b,
                        &mut scratch.borrow(),
                    );
                }
            }
        }

        if min_dnum < res_dnum {
            let mut res = res.to_backend_mut();
            for row in min_dnum..res_dnum {
                for col in 0..res_rank_in {
                    let mut ct = crate::layouts::gglwe_at_backend_mut_from_mut::<BE>(&mut res, row, col);
                    for data_col in 0..ct.data.cols() {
                        self.vec_znx_zero_backend(&mut ct.data, data_col);
                    }
                }
            }
        }
    }

    fn gglwe_external_product_inplace_default<'s, R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
        BE: 's,
    {
        assert_eq!(
            res.rank_out(),
            a.rank(),
            "res output rank: {} != a rank: {}",
            res.rank_out(),
            a.rank()
        );
        assert!(
            scratch.available() >= self.gglwe_external_product_tmp_bytes_default(res, res, a),
            "scratch.available(): {} < GGLWEExternalProduct::gglwe_external_product_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_external_product_tmp_bytes_default(res, res, a)
        );

        let res_dnum: usize = res.dnum().into();
        let res_rank_in: usize = res.rank_in().into();
        let mut res = res.to_backend_mut();
        for row in 0..res_dnum {
            for col in 0..res_rank_in {
                self.glwe_external_product_inplace(
                    &mut crate::layouts::gglwe_at_backend_mut_from_mut::<BE>(&mut res, row, col),
                    a,
                    &mut scratch.borrow(),
                );
            }
        }
    }
}

impl<BE: Backend> GGLWEExternalProductDefault<BE> for Module<BE> where Self: GLWEExternalProduct<BE> + VecZnxZeroBackend<BE> {}
