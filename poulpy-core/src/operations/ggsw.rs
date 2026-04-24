use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Module, ScratchArena},
};

pub use crate::api::GGSWRotate;
use crate::{
    GLWERotate, ScratchArenaTakeCore,
    layouts::{GGSWBackendMut, GGSWBackendRef, GGSWInfos, GLWEInfos, ggsw_at_backend_mut_from_mut, ggsw_at_backend_ref_from_ref},
};

#[doc(hidden)]
pub trait GGSWRotateDefault<BE: Backend>
where
    Self: GLWERotate<BE>,
{
    fn ggsw_rotate_tmp_bytes_default(&self) -> usize {
        self.glwe_rotate_tmp_bytes()
    }

    fn ggsw_rotate_default<'r, 'a>(&self, k: i64, res: &mut GGSWBackendMut<'r, BE>, a: &GGSWBackendRef<'a, BE>) {
        assert!(res.dnum() <= a.dnum());
        assert_eq!(res.dsize(), a.dsize());
        assert_eq!(res.rank(), a.rank());
        let rows: usize = res.dnum().into();
        let cols: usize = (res.rank() + 1).into();

        for row in 0..rows {
            for col in 0..cols {
                self.glwe_rotate(
                    k,
                    &mut ggsw_at_backend_mut_from_mut::<BE>(res, row, col),
                    &ggsw_at_backend_ref_from_ref::<BE>(a, row, col),
                );
            }
        }
    }

    fn ggsw_rotate_inplace_default<'s, 'r>(&self, k: i64, res: &mut GGSWBackendMut<'r, BE>, scratch: &mut ScratchArena<'s, BE>)
    where
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE> + ScratchAvailable,
    {
        assert!(
            scratch.available() >= self.ggsw_rotate_tmp_bytes_default(),
            "scratch.available(): {} < GGSWRotate::ggsw_rotate_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_rotate_tmp_bytes_default()
        );

        let rows: usize = res.dnum().into();
        let cols: usize = (res.rank() + 1).into();

        for row in 0..rows {
            for col in 0..cols {
                let mut scratch_iter = scratch.borrow();
                self.glwe_rotate_inplace(k, &mut ggsw_at_backend_mut_from_mut::<BE>(res, row, col), &mut scratch_iter);
            }
        }
    }
}

impl<BE: Backend> GGSWRotateDefault<BE> for Module<BE>
where
    Module<BE>: GLWERotate<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
}
