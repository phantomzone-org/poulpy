use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, Module, ScratchArena, ZnxZero},
};

pub use crate::api::GGSWExternalProduct;
use crate::{
    GLWEExternalProduct, ScratchArenaTakeCore,
    layouts::{GGSW, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GGSWToMut, GGSWToRef, prepared::GGSWPreparedToBackendRef},
};

#[doc(hidden)]
pub trait GGSWExternalProductDefault<BE: Backend>
where
    Self: GLWEExternalProduct<BE> + ModuleN,
{
    fn ggsw_external_product_tmp_bytes_default<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos,
    {
        self.glwe_external_product_tmp_bytes(res_infos, a_infos, b_infos)
    }

    fn ggsw_external_product_default<'s, R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToMut + GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToRef + GGSWToBackendRef<BE> + GGSWInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
        BE: 's,
    {
        assert_eq!(res.rank(), a.rank(), "res rank: {} != a rank: {}", res.rank(), a.rank());
        assert_eq!(res.rank(), b.rank(), "res rank: {} != b rank: {}", res.rank(), b.rank());
        assert_eq!(res.base2k(), a.base2k());
        assert!(
            scratch.available() >= self.ggsw_external_product_tmp_bytes_default(res, a, b),
            "scratch.available(): {} < GGSWExternalProduct::ggsw_external_product_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_external_product_tmp_bytes_default(res, a, b)
        );

        let min_dnum: usize = res.dnum().min(a.dnum()).into();
        let res_dnum: usize = res.dnum().into();
        let res_rank: usize = (res.rank() + 1).into();
        {
            let mut res = res.to_backend_mut();
            let a = a.to_backend_ref();
            for row in 0..min_dnum {
                for col in 0..res_rank {
                    self.glwe_external_product(
                        &mut crate::layouts::ggsw_at_backend_mut_from_mut::<BE>(&mut res, row, col),
                        &crate::layouts::ggsw_at_backend_ref_from_ref::<BE>(&a, row, col),
                        b,
                        &mut scratch.borrow(),
                    );
                }
            }
        }

        if min_dnum < res_dnum {
            let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
            for row in min_dnum..res_dnum {
                for col in 0..res_rank {
                    res.at_mut(row, col).data.zero();
                }
            }
        }
    }

    fn ggsw_external_product_inplace_default<'s, R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToMut + GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
        BE: 's,
    {
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.rank(), a.rank(), "res rank: {} != a rank: {}", res.rank(), a.rank());
        assert!(
            scratch.available() >= self.ggsw_external_product_tmp_bytes_default(res, res, a),
            "scratch.available(): {} < GGSWExternalProduct::ggsw_external_product_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_external_product_tmp_bytes_default(res, res, a)
        );

        let res_dnum: usize = res.dnum().into();
        let res_rank: usize = (res.rank() + 1).into();
        let mut res = res.to_backend_mut();
        for row in 0..res_dnum {
            for col in 0..res_rank {
                self.glwe_external_product_inplace(
                    &mut crate::layouts::ggsw_at_backend_mut_from_mut::<BE>(&mut res, row, col),
                    a,
                    &mut scratch.borrow(),
                );
            }
        }
    }
}

impl<BE: Backend> GGSWExternalProductDefault<BE> for Module<BE> where Self: GLWEExternalProduct<BE> + ModuleN {}
