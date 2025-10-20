use poulpy_hal::layouts::{Backend, DataMut, Module, Scratch};

use crate::{
    ScratchTakeCore,
    keyswitching::glwe_ct::GLWEKeyswitch,
    layouts::{
        AutomorphismKey, AutomorphismKeyToRef, GGLWE, GGLWEInfos, GGLWEToMut, GGLWEToRef, GLWESwitchingKey,
        GLWESwitchingKeyToRef,
        prepared::{GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedToRef},
    },
};

impl AutomorphismKey<Vec<u8>> {
    pub fn keyswitch_tmp_bytes<R, A, K, M, BE: Backend>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
        M: GGLWEKeyswitch<BE>,
    {
        module.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos)
    }
}

impl<DataSelf: DataMut> AutomorphismKey<DataSelf> {
    pub fn keyswitch<A, B, M, BE: Backend>(&mut self, module: &M, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        A: AutomorphismKeyToRef,
        B: GLWESwitchingKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
        M: GGLWEKeyswitch<BE>,
    {
        module.gglwe_keyswitch(&mut self.key.key, &a.to_ref().key.key, b, scratch);
    }

    pub fn keyswitch_inplace<A, M, BE: Backend>(&mut self, module: &M, a: &A, scratch: &mut Scratch<BE>)
    where
        A: GLWESwitchingKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
        M: GGLWEKeyswitch<BE>,
    {
        module.gglwe_keyswitch_inplace(&mut self.key.key, a, scratch);
    }
}

impl GLWESwitchingKey<Vec<u8>> {
    pub fn keyswitch_tmp_bytes<R, A, K, M, BE: Backend>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
        M: GGLWEKeyswitch<BE>,
    {
        module.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos)
    }
}

impl<DataSelf: DataMut> GLWESwitchingKey<DataSelf> {
    pub fn keyswitch<A, B, M, BE: Backend>(&mut self, module: &M, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        A: GLWESwitchingKeyToRef,
        B: GLWESwitchingKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
        M: GGLWEKeyswitch<BE>,
    {
        module.gglwe_keyswitch(&mut self.key, &a.to_ref().key, b, scratch);
    }

    pub fn keyswitch_inplace<A, M, BE: Backend>(&mut self, module: &M, a: &A, scratch: &mut Scratch<BE>)
    where
        A: GLWESwitchingKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
        M: GGLWEKeyswitch<BE>,
    {
        module.gglwe_keyswitch_inplace(&mut self.key, a, scratch);
    }
}

impl GGLWE<Vec<u8>> {
    pub fn keyswitch_tmp_bytes<R, A, K, M, BE: Backend>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
        M: GGLWEKeyswitch<BE>,
    {
        module.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos)
    }
}

impl<DataSelf: DataMut> GGLWE<DataSelf> {
    pub fn keyswitch<A, B, M, BE: Backend>(&mut self, module: &M, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        A: GGLWEToRef,
        B: GLWESwitchingKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
        M: GGLWEKeyswitch<BE>,
    {
        module.gglwe_keyswitch(self, a, b, scratch);
    }

    pub fn keyswitch_inplace<A, M, BE: Backend>(&mut self, module: &M, a: &A, scratch: &mut Scratch<BE>)
    where
        A: GLWESwitchingKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
        M: GGLWEKeyswitch<BE>,
    {
        module.gglwe_keyswitch_inplace(self, a, scratch);
    }
}

impl<BE: Backend> GGLWEKeyswitch<BE> for Module<BE> where Self: GLWEKeyswitch<BE> {}

pub trait GGLWEKeyswitch<BE: Backend>
where
    Self: GLWEKeyswitch<BE>,
{
    fn gglwe_keyswitch_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
    {
        self.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos)
    }

    fn gglwe_keyswitch<R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut,
        A: GGLWEToRef,
        B: GLWESwitchingKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GGLWE<&[u8]> = &a.to_ref();
        let b: &GLWESwitchingKeyPrepared<&[u8], BE> = &b.to_ref();

        assert_eq!(
            res.rank_in(),
            a.rank_in(),
            "res input rank: {} != a input rank: {}",
            res.rank_in(),
            a.rank_in()
        );
        assert_eq!(
            a.rank_out(),
            b.rank_in(),
            "res output rank: {} != b input rank: {}",
            a.rank_out(),
            b.rank_in()
        );
        assert_eq!(
            res.rank_out(),
            b.rank_out(),
            "res output rank: {} != b output rank: {}",
            res.rank_out(),
            b.rank_out()
        );
        assert!(
            res.dnum() <= a.dnum(),
            "res.dnum()={} > a.dnum()={}",
            res.dnum(),
            a.dnum()
        );
        assert_eq!(
            res.dsize(),
            a.dsize(),
            "res dsize: {} != a dsize: {}",
            res.dsize(),
            a.dsize()
        );

        for row in 0..res.dnum().into() {
            for col in 0..res.rank_in().into() {
                self.glwe_keyswitch(&mut res.at_mut(row, col), &a.at(row, col), b, scratch);
            }
        }
    }

    fn gglwe_keyswitch_inplace<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut,
        A: GLWESwitchingKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWESwitchingKeyPrepared<&[u8], BE> = &a.to_ref();

        assert_eq!(
            res.rank_out(),
            a.rank_out(),
            "res output rank: {} != a output rank: {}",
            res.rank_out(),
            a.rank_out()
        );

        for row in 0..res.dnum().into() {
            for col in 0..res.rank_in().into() {
                self.glwe_keyswitch_inplace(&mut res.at_mut(row, col), a, scratch);
            }
        }
    }
}

impl<DataSelf: DataMut> GLWESwitchingKey<DataSelf> {}
