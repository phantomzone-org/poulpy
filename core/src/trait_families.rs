use backend::hal::{
    api::{
        ScratchAvailable, SvpApply, SvpApplyInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare, TakeVecZnxDft,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAutomorphismInplace, VecZnxBigAddInplace, VecZnxBigAddNormal,
        VecZnxBigAddSmallInplace, VecZnxBigAlloc, VecZnxBigAllocBytes, VecZnxBigAutomorphismInplace, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallBInplace, VecZnxCopy, VecZnxDftAddInplace, VecZnxDftAlloc,
        VecZnxDftAllocBytes, VecZnxDftCopy, VecZnxDftFromVecZnx, VecZnxDftToVecZnxBigConsume, VecZnxDftToVecZnxBigTmpA,
        VecZnxFillUniform, VecZnxNegateInplace, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotate,
        VecZnxRotateInplace, VecZnxRshInplace, VecZnxSub, VecZnxSubABInplace, VmpApply, VmpApplyAdd, VmpApplyTmpBytes,
        VmpPMatAlloc, VmpPMatAllocBytes, VmpPMatPrepare,
    },
    layouts::{Backend, Module},
};

pub trait GGSWKeySwitchFamily<B> =
    GLWEKeyswitchFamily<B> + VecZnxBigAllocBytes + VecZnxDftCopy<B> + VecZnxDftAddInplace<B> + VecZnxDftToVecZnxBigTmpA<B>;
pub trait GGLWEEncryptSkFamily<B: Backend> = GLWEEncryptSkFamily<B>;
pub trait GGLWESwitchingKeyEncryptSkFamily<B: Backend> = GGLWEEncryptSkFamily<B>;
pub trait GGLWEAutomorphismKeyEncryptSkFamily<B: Backend> = GGLWEEncryptSkFamily<B>;
pub trait GGLWETensorKeyEncryptSkFamily<B: Backend> =
    GGLWEEncryptSkFamily<B> + VecZnxBigAllocBytes + VecZnxDftToVecZnxBigTmpA<B> + SvpApply<B>;
pub trait GGSWEncryptSkFamily<B: Backend> = GLWEEncryptSkFamily<B>;
pub trait GGSWAssertNoiseFamily<B: Backend> = GLWEDecryptFamily<B>
    + VecZnxBigAlloc<B>
    + VecZnxDftAlloc<B>
    + VecZnxBigNormalizeTmpBytes
    + VecZnxBigNormalize<B>
    + VecZnxDftToVecZnxBigTmpA<B>;
pub trait GLWEDecryptFamily<B: Backend> = VecZnxDftAllocBytes
    + VecZnxBigAllocBytes
    + VecZnxDftFromVecZnx<B>
    + SvpApplyInplace<B>
    + VecZnxDftToVecZnxBigConsume<B>
    + VecZnxBigAddInplace<B>
    + VecZnxBigAddSmallInplace<B>
    + VecZnxBigNormalize<B>
    + VecZnxNormalizeTmpBytes;
pub trait GLWEEncryptSkFamily<B: Backend> = VecZnxDftAllocBytes
    + VecZnxBigNormalize<B>
    + VecZnxDftFromVecZnx<B>
    + SvpApplyInplace<B>
    + VecZnxDftToVecZnxBigConsume<B>
    + VecZnxNormalizeTmpBytes
    + VecZnxFillUniform
    + VecZnxSubABInplace
    + VecZnxAddInplace
    + VecZnxNormalizeInplace<B>
    + VecZnxAddNormal
    + VecZnxNormalize<B>
    + VecZnxSub;
pub trait GLWEEncryptPkFamily<B: Backend> = VecZnxDftAllocBytes
    + VecZnxBigAllocBytes
    + SvpPPolAllocBytes
    + SvpPrepare<B>
    + SvpApply<B>
    + VecZnxDftToVecZnxBigConsume<B>
    + VecZnxBigAddNormal<B>
    + VecZnxBigAddSmallInplace<B>
    + VecZnxBigNormalize<B>
    + VecZnxNormalizeTmpBytes;
pub trait GLWEExternalProductFamily<B: Backend> = VecZnxDftAllocBytes
    + VmpApplyTmpBytes
    + VmpApply<B>
    + VmpApplyAdd<B>
    + VecZnxDftFromVecZnx<B>
    + VecZnxDftToVecZnxBigConsume<B>
    + VecZnxBigNormalize<B>
    + VecZnxNormalizeTmpBytes;
pub trait GLWEKeyswitchFamily<B: Backend> = VecZnxDftAllocBytes
    + VmpApplyTmpBytes
    + VecZnxBigNormalizeTmpBytes
    + VmpApplyTmpBytes
    + VmpApply<B>
    + VmpApplyAdd<B>
    + VecZnxDftFromVecZnx<B>
    + VecZnxDftToVecZnxBigConsume<B>
    + VecZnxBigAddSmallInplace<B>
    + VecZnxBigNormalize<B>;
pub trait GLWEPackingFamily<B: Backend> = GLWEKeyswitchFamily<B>
    + VecZnxCopy
    + VecZnxRotateInplace
    + VecZnxSub
    + VecZnxNegateInplace
    + VecZnxRshInplace
    + VecZnxAddInplace
    + VecZnxNormalizeInplace<B>
    + VecZnxSubABInplace
    + VecZnxRotate
    + VecZnxAutomorphismInplace
    + VecZnxBigSubSmallBInplace<B>
    + VecZnxBigAutomorphismInplace<B>;
pub trait GLWESecretExecModuleFamily<B: Backend> = SvpPrepare<B> + SvpPPolAllocBytes + SvpPPolAlloc<B>;
pub trait GGLWEExecLayoutFamily<B: Backend> = VmpPMatAlloc<B> + VmpPMatAllocBytes + VmpPMatPrepare<B>;
pub trait GLWETraceModuleFamily<B: Backend> =
    GLWEKeyswitchFamily<B> + VecZnxCopy + VecZnxRshInplace + VecZnxBigAutomorphismInplace<B>;
pub trait GLWETraceScratchFamily<B: Backend> = TakeVecZnxDft<B> + ScratchAvailable;
pub trait GGSWLayoutFamily<B: Backend> = VmpPMatAlloc<B> + VmpPMatAllocBytes + VmpPMatPrepare<B>;

pub trait DecompressFamily<B: Backend> = VecZnxFillUniform + VecZnxCopy;
pub trait Decompress<B: Backend, C> {
    fn decompress(&mut self, module: &Module<B>, other: &C)
    where
        Module<B>: DecompressFamily<B>;
}
