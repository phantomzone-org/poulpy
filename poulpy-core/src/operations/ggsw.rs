use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Module, Scratch},
};

use crate::{
    GLWERotate, ScratchTakeCore,
    layouts::{GGSW, GGSWInfos, GGSWToMut, GGSWToRef, GLWEInfos},
};

impl<BE: Backend> GGSWRotate<BE> for Module<BE> where Module<BE>: GLWERotate<BE> {}

pub trait GGSWRotate<BE: Backend>
where
    Self: GLWERotate<BE>,
{
    fn ggsw_rotate_tmp_bytes(&self) -> usize {
        self.glwe_rotate_tmp_bytes()
    }

    fn ggsw_rotate<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GGSWToMut,
        A: GGSWToRef,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let a: &GGSW<&[u8]> = &a.to_ref();

        assert!(res.dnum() <= a.dnum());
        assert_eq!(res.dsize(), a.dsize());
        assert_eq!(res.rank(), a.rank());
        let rows: usize = res.dnum().into();
        let cols: usize = (res.rank() + 1).into();

        for row in 0..rows {
            for col in 0..cols {
                self.glwe_rotate(k, &mut res.at_mut(row, col), &a.at(row, col));
            }
        }
    }

    fn ggsw_rotate_inplace<R>(&self, k: i64, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
    {
        assert!(
            scratch.available() >= self.ggsw_rotate_tmp_bytes(),
            "scratch.available(): {} < GGSWRotate::ggsw_rotate_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_rotate_tmp_bytes()
        );
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();

        let rows: usize = res.dnum().into();
        let cols: usize = (res.rank() + 1).into();

        for row in 0..rows {
            for col in 0..cols {
                self.glwe_rotate_inplace(k, &mut res.at_mut(row, col), scratch);
            }
        }
    }
}
