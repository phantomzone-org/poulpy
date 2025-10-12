use poulpy_hal::layouts::{Backend, DataMut, Scratch};

use crate::{
    ScratchTakeCore,
    external_product::gglwe_ksk::GGLWEExternalProduct,
    layouts::{AutomorphismKey, AutomorphismKeyToRef, GGLWEInfos, GGSWInfos, prepared::GGSWPreparedToRef},
};

impl AutomorphismKey<Vec<u8>> {
    pub fn external_product_tmp_bytes<R, A, B, M, BE: Backend>(
        &self,
        module: &M,
        res_infos: &R,
        a_infos: &A,
        b_infos: &B,
    ) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos,
        M: GGLWEExternalProduct<BE>,
    {
        module.gglwe_external_product_tmp_bytes(res_infos, a_infos, b_infos)
    }
}

impl<DataSelf: DataMut> AutomorphismKey<DataSelf> {
    pub fn external_product<A, B, M, BE: Backend>(&mut self, module: &M, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        M: GGLWEExternalProduct<BE>,
        A: AutomorphismKeyToRef,
        B: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.gglwe_external_product(&mut self.key.key, &a.to_ref().key.key, b, scratch);
    }

    pub fn external_product_inplace<A, M, BE: Backend>(&mut self, module: &M, a: &A, scratch: &mut Scratch<BE>)
    where
        M: GGLWEExternalProduct<BE>,
        A: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.gglwe_external_product_inplace(&mut self.key.key, a, scratch);
    }
}
