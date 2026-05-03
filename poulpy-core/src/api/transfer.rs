use poulpy_hal::layouts::{Backend, DataView, MatZnx, Module, ScalarZnx, SvpPPol, TransferFrom, VecZnx, VecZnxDft, VmpPMat};

use crate::layouts::{
    BackendGGLWE, BackendGGLWEPrepared, BackendGGLWEToGGSWKeyPrepared, BackendGGSW, BackendGGSWPrepared, BackendGLWE,
    BackendGLWEAutomorphismKeyPrepared, BackendGLWEPlaintext, BackendGLWEPrepared, BackendGLWEPublicKeyPrepared,
    BackendGLWESecret, BackendGLWESecretPrepared, BackendGLWESecretTensorPrepared, BackendGLWESwitchingKeyPrepared,
    BackendGLWETensorKeyPrepared, BackendGLWEToLWEKeyPrepared, BackendLWE, BackendLWEPlaintext, BackendLWESecret,
    BackendLWESwitchingKeyPrepared, BackendLWEToGLWEKeyPrepared, GGLWE, GGLWEPrepared, GGLWEToGGSWKeyPrepared, GGSW,
    GGSWPrepared, GLWE, GLWEAutomorphismKeyPrepared, GLWEPlaintext, GLWEPrepared, GLWEPublicKeyPrepared, GLWESecret,
    GLWESecretPrepared, GLWESecretTensorPrepared, GLWESwitchingKeyPrepared, GLWETensorKeyPrepared, GLWEToLWEKeyPrepared, LWE,
    LWEPlaintext, LWESecret, LWESwitchingKeyPrepared, LWEToGLWEKeyPrepared,
};

fn transfer_vec_znx<From, To>(src: &VecZnx<From::OwnedBuf>) -> VecZnx<To::OwnedBuf>
where
    From: Backend,
    To: Backend + TransferFrom<From>,
{
    VecZnx::from_data_with_max_size(
        <To as TransferFrom<From>>::transfer_buf(src.data()),
        src.n(),
        src.cols(),
        src.size(),
        src.max_size(),
    )
}

fn transfer_mat_znx<From, To>(src: &MatZnx<From::OwnedBuf>) -> MatZnx<To::OwnedBuf>
where
    From: Backend,
    To: Backend + TransferFrom<From>,
{
    MatZnx::from_data(
        <To as TransferFrom<From>>::transfer_buf(src.data()),
        src.n(),
        src.rows(),
        src.cols_in(),
        src.cols_out(),
        src.size(),
    )
}

fn transfer_scalar_znx<From, To>(src: &ScalarZnx<From::OwnedBuf>) -> ScalarZnx<To::OwnedBuf>
where
    From: Backend,
    To: Backend + TransferFrom<From>,
{
    ScalarZnx::from_data(<To as TransferFrom<From>>::transfer_buf(src.data()), src.n(), src.cols())
}

fn transfer_vec_znx_dft<From, To>(src: &VecZnxDft<From::OwnedBuf, From>) -> VecZnxDft<To::OwnedBuf, To>
where
    From: Backend,
    To: Backend + TransferFrom<From>,
{
    VecZnxDft::from_data(
        <To as TransferFrom<From>>::transfer_buf(src.data()),
        src.n(),
        src.cols(),
        src.size(),
    )
}

fn transfer_vmp_pmat<From, To>(src: &VmpPMat<From::OwnedBuf, From>) -> VmpPMat<To::OwnedBuf, To>
where
    From: Backend,
    To: Backend + TransferFrom<From>,
{
    VmpPMat::from_data(
        <To as TransferFrom<From>>::transfer_buf(src.data()),
        src.n(),
        src.rows(),
        src.cols_in(),
        src.cols_out(),
        src.size(),
    )
}

fn transfer_svp_ppol<From, To>(src: &SvpPPol<From::OwnedBuf, From>) -> SvpPPol<To::OwnedBuf, To>
where
    From: Backend,
    To: Backend + TransferFrom<From>,
{
    SvpPPol::from_data(<To as TransferFrom<From>>::transfer_buf(src.data()), src.n(), src.cols())
}

pub trait ModuleTransfer<To: Backend> {
    fn upload_glwe<From>(&self, src: &BackendGLWE<From>) -> BackendGLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_glwe<From>(&self, src: &BackendGLWE<From>) -> BackendGLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_lwe<From>(&self, src: &BackendLWE<From>) -> BackendLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_lwe<From>(&self, src: &BackendLWE<From>) -> BackendLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_gglwe<From>(&self, src: &BackendGGLWE<From>) -> BackendGGLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_gglwe<From>(&self, src: &BackendGGLWE<From>) -> BackendGGLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_ggsw<From>(&self, src: &BackendGGSW<From>) -> BackendGGSW<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_ggsw<From>(&self, src: &BackendGGSW<From>) -> BackendGGSW<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_glwe_secret<From>(&self, src: &BackendGLWESecret<From>) -> BackendGLWESecret<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_glwe_secret<From>(&self, src: &BackendGLWESecret<From>) -> BackendGLWESecret<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_lwe_secret<From>(&self, src: &BackendLWESecret<From>) -> BackendLWESecret<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_lwe_secret<From>(&self, src: &BackendLWESecret<From>) -> BackendLWESecret<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_glwe_plaintext<From>(&self, src: &BackendGLWEPlaintext<From>) -> BackendGLWEPlaintext<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_glwe_plaintext<From>(&self, src: &BackendGLWEPlaintext<From>) -> BackendGLWEPlaintext<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_lwe_plaintext<From>(&self, src: &BackendLWEPlaintext<From>) -> BackendLWEPlaintext<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_lwe_plaintext<From>(&self, src: &BackendLWEPlaintext<From>) -> BackendLWEPlaintext<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_glwe_prepared<From>(&self, src: &BackendGLWEPrepared<From>) -> BackendGLWEPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_glwe_prepared<From>(&self, src: &BackendGLWEPrepared<From>) -> BackendGLWEPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_gglwe_prepared<From>(&self, src: &BackendGGLWEPrepared<From>) -> BackendGGLWEPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_gglwe_prepared<From>(&self, src: &BackendGGLWEPrepared<From>) -> BackendGGLWEPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_ggsw_prepared<From>(&self, src: &BackendGGSWPrepared<From>) -> BackendGGSWPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_ggsw_prepared<From>(&self, src: &BackendGGSWPrepared<From>) -> BackendGGSWPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_glwe_secret_prepared<From>(&self, src: &BackendGLWESecretPrepared<From>) -> BackendGLWESecretPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_glwe_secret_prepared<From>(&self, src: &BackendGLWESecretPrepared<From>) -> BackendGLWESecretPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_glwe_public_key_prepared<From>(&self, src: &BackendGLWEPublicKeyPrepared<From>) -> BackendGLWEPublicKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_glwe_public_key_prepared<From>(
        &self,
        src: &BackendGLWEPublicKeyPrepared<From>,
    ) -> BackendGLWEPublicKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_glwe_secret_tensor_prepared<From>(
        &self,
        src: &BackendGLWESecretTensorPrepared<From>,
    ) -> BackendGLWESecretTensorPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_glwe_secret_tensor_prepared<From>(
        &self,
        src: &BackendGLWESecretTensorPrepared<From>,
    ) -> BackendGLWESecretTensorPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_glwe_switching_key_prepared<From>(
        &self,
        src: &BackendGLWESwitchingKeyPrepared<From>,
    ) -> BackendGLWESwitchingKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_glwe_switching_key_prepared<From>(
        &self,
        src: &BackendGLWESwitchingKeyPrepared<From>,
    ) -> BackendGLWESwitchingKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_glwe_automorphism_key_prepared<From>(
        &self,
        src: &BackendGLWEAutomorphismKeyPrepared<From>,
    ) -> BackendGLWEAutomorphismKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_glwe_automorphism_key_prepared<From>(
        &self,
        src: &BackendGLWEAutomorphismKeyPrepared<From>,
    ) -> BackendGLWEAutomorphismKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_glwe_tensor_key_prepared<From>(&self, src: &BackendGLWETensorKeyPrepared<From>) -> BackendGLWETensorKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_glwe_tensor_key_prepared<From>(
        &self,
        src: &BackendGLWETensorKeyPrepared<From>,
    ) -> BackendGLWETensorKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_glwe_to_lwe_key_prepared<From>(&self, src: &BackendGLWEToLWEKeyPrepared<From>) -> BackendGLWEToLWEKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_glwe_to_lwe_key_prepared<From>(&self, src: &BackendGLWEToLWEKeyPrepared<From>) -> BackendGLWEToLWEKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_lwe_switching_key_prepared<From>(
        &self,
        src: &BackendLWESwitchingKeyPrepared<From>,
    ) -> BackendLWESwitchingKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_lwe_switching_key_prepared<From>(
        &self,
        src: &BackendLWESwitchingKeyPrepared<From>,
    ) -> BackendLWESwitchingKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_lwe_to_glwe_key_prepared<From>(&self, src: &BackendLWEToGLWEKeyPrepared<From>) -> BackendLWEToGLWEKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_lwe_to_glwe_key_prepared<From>(&self, src: &BackendLWEToGLWEKeyPrepared<From>) -> BackendLWEToGLWEKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_gglwe_to_ggsw_key_prepared<From>(
        &self,
        src: &BackendGGLWEToGGSWKeyPrepared<From>,
    ) -> BackendGGLWEToGGSWKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_gglwe_to_ggsw_key_prepared<From>(
        &self,
        src: &BackendGGLWEToGGSWKeyPrepared<From>,
    ) -> BackendGGLWEToGGSWKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>;
}

impl<To: Backend> ModuleTransfer<To> for Module<To> {
    fn upload_glwe<From>(&self, src: &BackendGLWE<From>) -> BackendGLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        let _ = self;
        GLWE {
            data: transfer_vec_znx::<From, To>(&src.data),
            base2k: src.base2k,
        }
    }

    fn download_glwe<From>(&self, src: &BackendGLWE<From>) -> BackendGLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_glwe(src)
    }

    fn upload_lwe<From>(&self, src: &BackendLWE<From>) -> BackendLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        let _ = self;
        LWE {
            data: transfer_vec_znx::<From, To>(&src.data),
            base2k: src.base2k,
        }
    }

    fn download_lwe<From>(&self, src: &BackendLWE<From>) -> BackendLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_lwe(src)
    }

    fn upload_gglwe<From>(&self, src: &BackendGGLWE<From>) -> BackendGGLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        let _ = self;
        GGLWE {
            data: transfer_mat_znx::<From, To>(&src.data),
            base2k: src.base2k,
            dsize: src.dsize,
        }
    }

    fn download_gglwe<From>(&self, src: &BackendGGLWE<From>) -> BackendGGLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_gglwe(src)
    }

    fn upload_ggsw<From>(&self, src: &BackendGGSW<From>) -> BackendGGSW<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        let _ = self;
        GGSW {
            data: transfer_mat_znx::<From, To>(&src.data),
            base2k: src.base2k,
            dsize: src.dsize,
        }
    }

    fn download_ggsw<From>(&self, src: &BackendGGSW<From>) -> BackendGGSW<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_ggsw(src)
    }

    fn upload_glwe_secret<From>(&self, src: &BackendGLWESecret<From>) -> BackendGLWESecret<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        let _ = self;
        GLWESecret {
            data: transfer_scalar_znx::<From, To>(&src.data),
            dist: src.dist,
        }
    }

    fn download_glwe_secret<From>(&self, src: &BackendGLWESecret<From>) -> BackendGLWESecret<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_glwe_secret(src)
    }

    fn upload_lwe_secret<From>(&self, src: &BackendLWESecret<From>) -> BackendLWESecret<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        let _ = self;
        LWESecret {
            data: transfer_scalar_znx::<From, To>(&src.data),
            dist: src.dist,
        }
    }

    fn download_lwe_secret<From>(&self, src: &BackendLWESecret<From>) -> BackendLWESecret<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_lwe_secret(src)
    }

    fn upload_glwe_plaintext<From>(&self, src: &BackendGLWEPlaintext<From>) -> BackendGLWEPlaintext<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        let _ = self;
        GLWEPlaintext {
            data: transfer_vec_znx::<From, To>(&src.data),
            base2k: src.base2k,
        }
    }

    fn download_glwe_plaintext<From>(&self, src: &BackendGLWEPlaintext<From>) -> BackendGLWEPlaintext<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_glwe_plaintext(src)
    }

    fn upload_lwe_plaintext<From>(&self, src: &BackendLWEPlaintext<From>) -> BackendLWEPlaintext<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        let _ = self;
        LWEPlaintext {
            data: transfer_vec_znx::<From, To>(&src.data),
            base2k: src.base2k,
        }
    }

    fn download_lwe_plaintext<From>(&self, src: &BackendLWEPlaintext<From>) -> BackendLWEPlaintext<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_lwe_plaintext(src)
    }

    fn upload_glwe_prepared<From>(&self, src: &BackendGLWEPrepared<From>) -> BackendGLWEPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        let _ = self;
        GLWEPrepared {
            data: transfer_vec_znx_dft::<From, To>(&src.data),
            base2k: src.base2k,
        }
    }

    fn download_glwe_prepared<From>(&self, src: &BackendGLWEPrepared<From>) -> BackendGLWEPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_glwe_prepared(src)
    }

    fn upload_gglwe_prepared<From>(&self, src: &BackendGGLWEPrepared<From>) -> BackendGGLWEPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        let _ = self;
        GGLWEPrepared {
            data: transfer_vmp_pmat::<From, To>(&src.data),
            base2k: src.base2k,
            dsize: src.dsize,
        }
    }

    fn download_gglwe_prepared<From>(&self, src: &BackendGGLWEPrepared<From>) -> BackendGGLWEPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_gglwe_prepared(src)
    }

    fn upload_ggsw_prepared<From>(&self, src: &BackendGGSWPrepared<From>) -> BackendGGSWPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        let _ = self;
        GGSWPrepared {
            data: transfer_vmp_pmat::<From, To>(&src.data),
            base2k: src.base2k,
            dsize: src.dsize,
        }
    }

    fn download_ggsw_prepared<From>(&self, src: &BackendGGSWPrepared<From>) -> BackendGGSWPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_ggsw_prepared(src)
    }

    fn upload_glwe_secret_prepared<From>(&self, src: &BackendGLWESecretPrepared<From>) -> BackendGLWESecretPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        let _ = self;
        GLWESecretPrepared {
            data: transfer_svp_ppol::<From, To>(&src.data),
            dist: src.dist,
        }
    }

    fn download_glwe_secret_prepared<From>(&self, src: &BackendGLWESecretPrepared<From>) -> BackendGLWESecretPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_glwe_secret_prepared(src)
    }

    fn upload_glwe_public_key_prepared<From>(&self, src: &BackendGLWEPublicKeyPrepared<From>) -> BackendGLWEPublicKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        GLWEPublicKeyPrepared {
            key: self.upload_glwe_prepared::<From>(&src.key),
            dist: src.dist,
        }
    }

    fn download_glwe_public_key_prepared<From>(
        &self,
        src: &BackendGLWEPublicKeyPrepared<From>,
    ) -> BackendGLWEPublicKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_glwe_public_key_prepared(src)
    }

    fn upload_glwe_secret_tensor_prepared<From>(
        &self,
        src: &BackendGLWESecretTensorPrepared<From>,
    ) -> BackendGLWESecretTensorPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        let _ = self;
        GLWESecretTensorPrepared {
            data: transfer_svp_ppol::<From, To>(&src.data),
            rank: src.rank,
            dist: src.dist,
        }
    }

    fn download_glwe_secret_tensor_prepared<From>(
        &self,
        src: &BackendGLWESecretTensorPrepared<From>,
    ) -> BackendGLWESecretTensorPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_glwe_secret_tensor_prepared(src)
    }

    fn upload_glwe_switching_key_prepared<From>(
        &self,
        src: &BackendGLWESwitchingKeyPrepared<From>,
    ) -> BackendGLWESwitchingKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        GLWESwitchingKeyPrepared {
            key: self.upload_gglwe_prepared::<From>(&src.key),
            input_degree: src.input_degree,
            output_degree: src.output_degree,
        }
    }

    fn download_glwe_switching_key_prepared<From>(
        &self,
        src: &BackendGLWESwitchingKeyPrepared<From>,
    ) -> BackendGLWESwitchingKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_glwe_switching_key_prepared(src)
    }

    fn upload_glwe_automorphism_key_prepared<From>(
        &self,
        src: &BackendGLWEAutomorphismKeyPrepared<From>,
    ) -> BackendGLWEAutomorphismKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        GLWEAutomorphismKeyPrepared {
            key: self.upload_gglwe_prepared::<From>(&src.key),
            p: src.p,
        }
    }

    fn download_glwe_automorphism_key_prepared<From>(
        &self,
        src: &BackendGLWEAutomorphismKeyPrepared<From>,
    ) -> BackendGLWEAutomorphismKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_glwe_automorphism_key_prepared(src)
    }

    fn upload_glwe_tensor_key_prepared<From>(&self, src: &BackendGLWETensorKeyPrepared<From>) -> BackendGLWETensorKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        GLWETensorKeyPrepared(self.upload_gglwe_prepared::<From>(&src.0))
    }

    fn download_glwe_tensor_key_prepared<From>(
        &self,
        src: &BackendGLWETensorKeyPrepared<From>,
    ) -> BackendGLWETensorKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_glwe_tensor_key_prepared(src)
    }

    fn upload_glwe_to_lwe_key_prepared<From>(&self, src: &BackendGLWEToLWEKeyPrepared<From>) -> BackendGLWEToLWEKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        GLWEToLWEKeyPrepared(self.upload_glwe_switching_key_prepared::<From>(&src.0))
    }

    fn download_glwe_to_lwe_key_prepared<From>(&self, src: &BackendGLWEToLWEKeyPrepared<From>) -> BackendGLWEToLWEKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_glwe_to_lwe_key_prepared(src)
    }

    fn upload_lwe_switching_key_prepared<From>(
        &self,
        src: &BackendLWESwitchingKeyPrepared<From>,
    ) -> BackendLWESwitchingKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        LWESwitchingKeyPrepared(self.upload_glwe_switching_key_prepared::<From>(&src.0))
    }

    fn download_lwe_switching_key_prepared<From>(
        &self,
        src: &BackendLWESwitchingKeyPrepared<From>,
    ) -> BackendLWESwitchingKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_lwe_switching_key_prepared(src)
    }

    fn upload_lwe_to_glwe_key_prepared<From>(&self, src: &BackendLWEToGLWEKeyPrepared<From>) -> BackendLWEToGLWEKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        LWEToGLWEKeyPrepared(self.upload_glwe_switching_key_prepared::<From>(&src.0))
    }

    fn download_lwe_to_glwe_key_prepared<From>(&self, src: &BackendLWEToGLWEKeyPrepared<From>) -> BackendLWEToGLWEKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_lwe_to_glwe_key_prepared(src)
    }

    fn upload_gglwe_to_ggsw_key_prepared<From>(
        &self,
        src: &BackendGGLWEToGGSWKeyPrepared<From>,
    ) -> BackendGGLWEToGGSWKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        GGLWEToGGSWKeyPrepared {
            keys: src.keys.iter().map(|key| self.upload_gglwe_prepared::<From>(key)).collect(),
        }
    }

    fn download_gglwe_to_ggsw_key_prepared<From>(
        &self,
        src: &BackendGGLWEToGGSWKeyPrepared<From>,
    ) -> BackendGGLWEToGGSWKeyPrepared<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_gglwe_to_ggsw_key_prepared(src)
    }
}
