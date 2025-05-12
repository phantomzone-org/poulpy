use base2k::{Backend, Module, Scratch};
use sampling::source::Source;

pub trait EncryptSkScratchSpace {
    fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, ct_size: usize) -> usize;
}

pub trait EncryptSk<DataCt, DataPt, DataSk, B: Backend> {
    type Ciphertext;
    type Plaintext;
    type SecretKey;

    fn encrypt_sk(
        &self,
        module: &Module<B>,
        ct: &mut Self::Ciphertext,
        pt: &Self::Plaintext,
        sk: &Self::SecretKey,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        bound: f64,
        scratch: &mut Scratch,
    );
}

pub trait EncryptZeroSkScratchSpace {
    fn encrypt_zero_sk_scratch_space<B: Backend>(module: &Module<B>, ct_size: usize) -> usize;
}

pub trait EncryptZeroSk<DatCt, DataSk, B: Backend> {
    type Ciphertext;
    type SecretKey;

    fn encrypt_zero_sk(
        &self,
        module: &Module<B>,
        ct: &mut Self::Ciphertext,
        sk: &Self::SecretKey,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        bound: f64,
        scratch: &mut Scratch,
    );
}

pub trait EncryptPkScratchSpace {
    fn encrypt_pk_scratch_space<B: Backend>(module: &Module<B>, ct_size: usize) -> usize;
}

pub trait EncryptPk<DataCt, DataPt, DataPk, B: Backend> {
    type Ciphertext;
    type Plaintext;
    type PublicKey;

    fn encrypt_pk(
        &self,
        module: &Module<B>,
        ct: &mut Self::Ciphertext,
        pt: &Self::Plaintext,
        pk: &Self::PublicKey,
        source_xu: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        bound: f64,
        scratch: &mut Scratch,
    );
}

pub trait EncryptZeroPkScratchSpace {
    fn encrypt_zero_pk_scratch_space<B: Backend>(module: &Module<B>, ct_size: usize) -> usize;
}

pub trait EncryptZeroPk<DataCt, DataPk, B: Backend> {
    type Ciphertext;
    type PublicKey;

    fn encrypt_zero_pk(
        &self,
        module: &Module<B>,
        ct: &mut Self::Ciphertext,
        pk: &Self::PublicKey,
        source_xu: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        bound: f64,
        scratch: &mut Scratch,
    );
}

pub trait Decrypt<DataPt, DataCt, DataSk, B: Backend> {
    type Plaintext;
    type Ciphertext;
    type SecretKey;

    fn decrypt(
        &self,
        module: &Module<B>,
        pt: &mut Self::Plaintext,
        ct: &Self::Ciphertext,
        sk: &Self::SecretKey,
        scratch: &mut Scratch,
    );
}
