use poulpy_core::{
    GLWECopy, GLWERotate, ScratchArenaTakeCore,
    layouts::{
        GGSW, GGSWInfos, GGSWToMut, GGSWToRef, GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GLWEToRef, LWEInfos,
        glwe_backend_mut_from_mut, glwe_backend_ref_from_mut, glwe_backend_ref_from_ref,
    },
};
use poulpy_hal::{
    api::{VecZnxAddScalarAssign, VecZnxNormalizeInplace},
    layouts::{Backend, HostDataMut, Module, ScalarZnx, ScalarZnxToRef, ScratchArena, ZnxZero},
};

use crate::bdd_arithmetic::{Cmux, GetGGSWBit, UnsignedInteger};

impl<T: UnsignedInteger, BE: Backend<OwnedBuf = Vec<u8>>> GGSWBlindRotation<T, BE> for Module<BE>
where
    Self: GLWEBlindRotation<BE> + VecZnxAddScalarAssign + VecZnxNormalizeInplace<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
}

/// Extends [`GLWEBlindRotation`] to lift blind rotation to GGSW matrices and
/// to construct a GGSW from a scalar test-vector via blind rotation.
///
/// A GGSW matrix is a `(dnum × (rank+1))` array of GLWE ciphertexts.  The two
/// methods in this trait apply [`GLWEBlindRotation`] row-by-row:
///
/// - `ggsw_blind_rotation`: rotates each GLWE row of an existing GGSW by the
///   encrypted exponent derived from `fhe_uint`.
/// - `scalar_to_ggsw_blind_rotation`: constructs a fresh GGSW by first placing
///   the scalar test-vector into each row of a temporary GLWE and then rotating.
pub trait GGSWBlindRotation<T: UnsignedInteger, BE: Backend<OwnedBuf = Vec<u8>>>
where
    Self: GLWEBlindRotation<BE> + VecZnxAddScalarAssign + VecZnxNormalizeAssign<BE>,
{
    /// Returns the minimum scratch-space size in bytes required by
    /// [`ggsw_blind_rotation`][Self::ggsw_blind_rotation].
    fn ggsw_to_ggsw_blind_rotation_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        self.glwe_blind_rotation_tmp_bytes(res_infos, k_infos)
    }

    #[allow(clippy::too_many_arguments)]
    /// res <- res * X^{((k>>bit_rsh) % 2^bit_mask) << bit_lsh}.
    fn ggsw_blind_rotation_assign<R, K>(
        &self,
        res: &mut R,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToMut,
        K: GetGGSWBit<BE>,
        BE: Backend<OwnedBuf = Vec<u8>>,
        BE: 'static,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();

        for col in 0..(res.rank() + 1).into() {
            for row in 0..res.dnum().into() {
                let mut res_at = res.at_mut(row, col);
                let mut res_at_backend = glwe_backend_mut_from_mut::<BE>(&mut res_at);
                self.glwe_blind_rotation_inplace(&mut res_at_backend, fhe_uint, sign, bit_rsh, bit_mask, bit_lsh, scratch);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// res <- a * X^{((k>>bit_rsh) % 2^bit_mask) << bit_lsh}.
    fn ggsw_blind_rotation<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToMut,
        A: GGSWToRef,
        K: GetGGSWBit<BE>,
        BE: Backend<OwnedBuf = Vec<u8>>,
        BE: 'static,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let a: &GGSW<&[u8]> = &a.to_ref();

        assert!(res.dnum() <= a.dnum());
        assert_eq!(res.dsize(), a.dsize());

        for col in 0..(res.rank() + 1).into() {
            for row in 0..res.dnum().into() {
                let mut res_at = res.at_mut(row, col);
                let a_at = a.at(row, col);
                self.glwe_blind_rotation(
                    &mut glwe_backend_mut_from_mut::<BE>(&mut res_at),
                    &glwe_backend_ref_from_ref::<BE>(&a_at),
                    fhe_uint,
                    sign,
                    bit_rsh,
                    bit_mask,
                    bit_lsh,
                    scratch,
                );
            }
        }
    }

    fn scalar_to_ggsw_blind_rotation_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        self.glwe_blind_rotation_tmp_bytes(res_infos, k_infos) + GLWE::<Vec<u8>>::bytes_of_from_infos(res_infos)
    }

    #[allow(clippy::too_many_arguments)]
    fn scalar_to_ggsw_blind_rotation<R, S, K>(
        &self,
        res: &mut R,
        test_vector: &S,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToMut,
        S: ScalarZnxToRef,
        K: GetGGSWBit<BE>,
        BE: Backend<OwnedBuf = Vec<u8>>,
        BE: 'static,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let test_vector: &ScalarZnx<&[u8]> = &test_vector.to_ref();

        let base2k: usize = res.base2k().into();
        let dsize: usize = res.dsize().into();

        // TODO(device): this helper still stages a host-owned GLWE row before
        // calling backend-generic blind rotation.
        let mut tmp_glwe: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&*res);
        let mut scratch_1 = scratch.borrow();

        for col in 0..(res.rank() + 1).into() {
            for row in 0..res.dnum().into() {
                tmp_glwe.data_mut().zero();
                self.vec_znx_add_scalar_assign(tmp_glwe.data_mut(), col, (dsize - 1) + row * dsize, test_vector, 0);
                self.vec_znx_normalize_inplace(base2k, tmp_glwe.data_mut(), col, &mut scratch_1.borrow());

                let mut res_at = res.at_mut(row, col);
                self.glwe_blind_rotation(
                    &mut glwe_backend_mut_from_mut::<BE>(&mut res_at),
                    &<GLWE<Vec<u8>> as GLWEToBackendRef<BE>>::to_backend_ref(&tmp_glwe),
                    fhe_uint,
                    sign,
                    bit_rsh,
                    bit_mask,
                    bit_lsh,
                    &mut scratch_1.borrow(),
                );
            }
        }
    }
}

impl<BE: Backend<OwnedBuf = Vec<u8>>> GLWEBlindRotation<BE> for Module<BE>
where
    Self: GLWECopy<BE> + GLWERotate<BE> + Cmux<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
}

/// Homomorphic rotation of a GLWE ciphertext by an encrypted exponent.
///
/// Given a GLWE ciphertext `a` and a set of GGSW ciphertexts encoding the bits
/// of an integer `k`, computes:
///
/// ```text
/// res = a * X^{sign * ((k >> bit_rsh) % 2^bit_mask) << bit_lsh}
/// ```
///
/// where `sign` controls whether the rotation is positive or negative.
/// The operation is performed using `bit_mask` successive CMux gates, one per
/// bit of the shift amount.
pub trait GLWEBlindRotation<BE: Backend<OwnedBuf = Vec<u8>>>
where
    Self: GLWECopy<BE> + GLWERotate<BE> + Cmux<BE>,
{
    /// Returns the minimum scratch-space size in bytes required by
    /// [`glwe_blind_rotation`][Self::glwe_blind_rotation].
    fn glwe_blind_rotation_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        self.cmux_tmp_bytes(res_infos, res_infos, k_infos) + GLWE::<Vec<u8>>::bytes_of_from_infos(res_infos)
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_rotation_inplace<K>(
        &self,
        res: &mut poulpy_core::layouts::GLWEBackendMut<'_, BE>,
        value: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        K: GetGGSWBit<BE>,
        BE: Backend<OwnedBuf = Vec<u8>>,
        BE: 'static,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
    {
        let res_infos = res.glwe_layout();
        // TODO(device): this ping-pong helper still relies on a host-owned
        // temporary ciphertexts for both ping-pong branches.
        let mut res_cur: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&res_infos);
        let mut tmp_res: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&res_infos);
        let mut scratch_1 = scratch.borrow();
        self.glwe_copy(
            &mut <GLWE<Vec<u8>> as GLWEToBackendMut<BE>>::to_backend_mut(&mut res_cur),
            &glwe_backend_ref_from_mut::<BE>(res),
        );

        // a_is_cur = true  => source is `res_cur`, dest is `tmp_res`
        // a_is_cur = false => source is `tmp_res`, dest is `res_cur`
        let mut a_is_cur: bool = true;

        for i in 0..bit_mask {
            if a_is_cur {
                let res_cur_ref = res_cur.to_ref();
                let res_cur_ref_backend = <GLWE<Vec<u8>> as GLWEToBackendRef<BE>>::to_backend_ref(&res_cur);
                let mut tmp_res_backend = <GLWE<Vec<u8>> as GLWEToBackendMut<BE>>::to_backend_mut(&mut tmp_res);
                match sign {
                    true => self.glwe_rotate(1 << (i + bit_lsh), &mut tmp_res_backend, &res_cur_ref_backend),
                    false => self.glwe_rotate(-1 << (i + bit_lsh), &mut tmp_res_backend, &res_cur_ref_backend),
                }

                let bit = value.get_bit(i + bit_rsh);
                self.cmux_inplace(&mut tmp_res_backend, &res_cur_ref, bit, &mut scratch_1.borrow());
            } else {
                let tmp_res_ref = tmp_res.to_ref();
                let tmp_res_ref_backend = <GLWE<Vec<u8>> as GLWEToBackendRef<BE>>::to_backend_ref(&tmp_res);
                let mut res_cur_backend = <GLWE<Vec<u8>> as GLWEToBackendMut<BE>>::to_backend_mut(&mut res_cur);
                match sign {
                    true => self.glwe_rotate(1 << (i + bit_lsh), &mut res_cur_backend, &tmp_res_ref_backend),
                    false => self.glwe_rotate(-1 << (i + bit_lsh), &mut res_cur_backend, &tmp_res_ref_backend),
                }

                let bit = value.get_bit(i + bit_rsh);
                self.cmux_inplace(&mut res_cur_backend, &tmp_res_ref, bit, &mut scratch_1.borrow());
            }

            // ping-pong roles for next iter
            a_is_cur = !a_is_cur;
        }

        let final_res: &GLWE<Vec<u8>> = if a_is_cur { &res_cur } else { &tmp_res };
        self.glwe_copy(res, &<GLWE<Vec<u8>> as GLWEToBackendRef<BE>>::to_backend_ref(final_res));
    }

    #[allow(clippy::too_many_arguments)]
    /// res <- a * X^{sign * ((k>>bit_rsh) % 2^bit_mask) << bit_lsh}.
    fn glwe_blind_rotation<K>(
        &self,
        res: &mut poulpy_core::layouts::GLWEBackendMut<'_, BE>,
        a: &poulpy_core::layouts::GLWEBackendRef<'_, BE>,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        K: GetGGSWBit<BE>,
        BE: Backend<OwnedBuf = Vec<u8>>,
        BE: 'static,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
    {
        self.glwe_copy(res, a);
        self.glwe_blind_rotation_assign(res, fhe_uint, sign, bit_rsh, bit_mask, bit_lsh, scratch);
    }
}
