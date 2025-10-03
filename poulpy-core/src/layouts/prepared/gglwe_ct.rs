use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VmpPMat, ZnxInfos},
    oep::VmpPMatAllocBytesImpl,
};

use crate::layouts::{
    Base2K, BuildError, Degree, Digits, GGLWECiphertext, GGLWEInfos, GLWEInfos, LWEInfos, Rank, Rows, TorusPrecision,
    prepared::{Prepare, PrepareAlloc},
};

#[derive(PartialEq, Eq)]
pub struct GGLWECiphertextPrepared<D: Data, B: Backend> {
    pub(crate) data: VmpPMat<D, B>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) digits: Digits,
}

impl<D: Data, B: Backend> LWEInfos for GGLWECiphertextPrepared<D, B> {
    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

impl<D: Data, B: Backend> GLWEInfos for GGLWECiphertextPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for GGLWECiphertextPrepared<D, B> {
    fn rank_in(&self) -> Rank {
        Rank(self.data.cols_in() as u32)
    }

    fn rank_out(&self) -> Rank {
        Rank(self.data.cols_out() as u32 - 1)
    }

    fn digits(&self) -> Digits {
        self.digits
    }

    fn rows(&self) -> Rows {
        Rows(self.data.rows() as u32)
    }
}

pub struct GGLWECiphertextPreparedBuilder<D: Data, B: Backend> {
    data: Option<VmpPMat<D, B>>,
    base2k: Option<Base2K>,
    k: Option<TorusPrecision>,
    digits: Option<Digits>,
}

impl<D: Data, B: Backend> GGLWECiphertextPrepared<D, B> {
    #[inline]
    pub fn builder() -> GGLWECiphertextPreparedBuilder<D, B> {
        GGLWECiphertextPreparedBuilder {
            data: None,
            base2k: None,
            k: None,
            digits: None,
        }
    }
}

impl<B: Backend> GGLWECiphertextPreparedBuilder<Vec<u8>, B> {
    #[inline]
    pub fn layout<A>(mut self, infos: &A) -> Self
    where
        A: GGLWEInfos,
        B: VmpPMatAllocBytesImpl<B>,
    {
        self.data = Some(VmpPMat::alloc(
            infos.n().into(),
            infos.rows().into(),
            infos.rank_in().into(),
            (infos.rank_out() + 1).into(),
            infos.size(),
        ));
        self.base2k = Some(infos.base2k());
        self.k = Some(infos.k());
        self.digits = Some(infos.digits());
        self
    }
}

impl<D: Data, B: Backend> GGLWECiphertextPreparedBuilder<D, B> {
    #[inline]
    pub fn data(mut self, data: VmpPMat<D, B>) -> Self {
        self.data = Some(data);
        self
    }
    #[inline]
    pub fn base2k(mut self, base2k: Base2K) -> Self {
        self.base2k = Some(base2k);
        self
    }
    #[inline]
    pub fn k(mut self, k: TorusPrecision) -> Self {
        self.k = Some(k);
        self
    }

    #[inline]
    pub fn digits(mut self, digits: Digits) -> Self {
        self.digits = Some(digits);
        self
    }

    pub fn build(self) -> Result<GGLWECiphertextPrepared<D, B>, BuildError> {
        let data: VmpPMat<D, B> = self.data.ok_or(BuildError::MissingData)?;
        let base2k: Base2K = self.base2k.ok_or(BuildError::MissingBase2K)?;
        let k: TorusPrecision = self.k.ok_or(BuildError::MissingK)?;
        let digits: Digits = self.digits.ok_or(BuildError::MissingDigits)?;

        if base2k == 0_u32 {
            return Err(BuildError::ZeroBase2K);
        }

        if digits == 0_u32 {
            return Err(BuildError::ZeroBase2K);
        }

        if k == 0_u32 {
            return Err(BuildError::ZeroTorusPrecision);
        }

        if data.n() == 0 {
            return Err(BuildError::ZeroDegree);
        }

        if data.cols() == 0 {
            return Err(BuildError::ZeroCols);
        }

        if data.size() == 0 {
            return Err(BuildError::ZeroLimbs);
        }

        Ok(GGLWECiphertextPrepared {
            data,
            base2k,
            k,
            digits,
        })
    }
}

impl<B: Backend> GGLWECiphertextPrepared<Vec<u8>, B> {
    pub fn alloc<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAlloc<B>,
    {
        debug_assert_eq!(module.n(), infos.n().0 as usize, "module.n() != infos.n()");
        Self::alloc_with(
            module,
            infos.base2k(),
            infos.k(),
            infos.rows(),
            infos.digits(),
            infos.rank_in(),
            infos.rank_out(),
        )
    }

    pub fn alloc_with(
        module: &Module<B>,
        base2k: Base2K,
        k: TorusPrecision,
        rows: Rows,
        digits: Digits,
        rank_in: Rank,
        rank_out: Rank,
    ) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        debug_assert!(
            size as u32 > digits.0,
            "invalid gglwe: ceil(k/base2k): {size} <= digits: {}",
            digits.0
        );

        assert!(
            rows.0 * digits.0 <= size as u32,
            "invalid gglwe: rows: {} * digits:{} > ceil(k/base2k): {size}",
            rows.0,
            digits.0,
        );

        Self {
            data: module.vmp_pmat_alloc(rows.into(), rank_in.into(), (rank_out + 1).into(), size),
            k,
            base2k,
            digits,
        }
    }

    pub fn alloc_bytes<A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAllocBytes,
    {
        debug_assert_eq!(module.n(), infos.n().0 as usize, "module.n() != infos.n()");
        Self::alloc_bytes_with(
            module,
            infos.base2k(),
            infos.k(),
            infos.rows(),
            infos.digits(),
            infos.rank_in(),
            infos.rank_out(),
        )
    }

    pub fn alloc_bytes_with(
        module: &Module<B>,
        base2k: Base2K,
        k: TorusPrecision,
        rows: Rows,
        digits: Digits,
        rank_in: Rank,
        rank_out: Rank,
    ) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        debug_assert!(
            size as u32 > digits.0,
            "invalid gglwe: ceil(k/base2k): {size} <= digits: {}",
            digits.0
        );

        assert!(
            rows.0 * digits.0 <= size as u32,
            "invalid gglwe: rows: {} * digits:{} > ceil(k/base2k): {size}",
            rows.0,
            digits.0,
        );

        module.vmp_pmat_alloc_bytes(rows.into(), rank_in.into(), (rank_out + 1).into(), size)
    }
}

impl<D: DataMut, DR: DataRef, B: Backend> Prepare<B, GGLWECiphertext<DR>> for GGLWECiphertextPrepared<D, B>
where
    Module<B>: VmpPrepare<B>,
{
    fn prepare(&mut self, module: &Module<B>, other: &GGLWECiphertext<DR>, scratch: &mut Scratch<B>) {
        module.vmp_prepare(&mut self.data, &other.data, scratch);
        self.k = other.k;
        self.base2k = other.base2k;
        self.digits = other.digits;
    }
}

impl<D: DataRef, B: Backend> PrepareAlloc<B, GGLWECiphertextPrepared<Vec<u8>, B>> for GGLWECiphertext<D>
where
    Module<B>: VmpPMatAlloc<B> + VmpPrepare<B>,
{
    fn prepare_alloc(&self, module: &Module<B>, scratch: &mut Scratch<B>) -> GGLWECiphertextPrepared<Vec<u8>, B> {
        let mut atk_prepared: GGLWECiphertextPrepared<Vec<u8>, B> = GGLWECiphertextPrepared::alloc(module, self);
        atk_prepared.prepare(module, self, scratch);
        atk_prepared
    }
}
