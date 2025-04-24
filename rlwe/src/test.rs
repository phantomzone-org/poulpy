use base2k::{alloc_aligned, SvpPPol, SvpPPolOps, VecZnx, BACKEND};
use sampling::source::{Source, new_seed};
use crate::{ciphertext::Ciphertext, decryptor::decrypt_rlwe, elem::ElemCommon, encryptor::encrypt_rlwe_sk, keys::SecretKey, parameters::{Parameters, ParametersLiteral, DEFAULT_SIGMA}, plaintext::Plaintext};



pub struct Context{
    pub params: Parameters,
    pub sk0: SecretKey,
    pub sk0_ppol:SvpPPol,
    pub sk1: SecretKey,
    pub sk1_ppol: SvpPPol,
    pub tmp_bytes: Vec<u8>,
}

impl Context{
    pub fn new(log_n: usize, log_base2k: usize, log_q: usize, log_p: usize) -> Self{

        let params_lit: ParametersLiteral = ParametersLiteral {
            backend: BACKEND::FFT64,
            log_n: log_n,
            log_q: log_q,
            log_p: log_p,
            log_base2k: log_base2k,
            log_scale: 20,
            xe: DEFAULT_SIGMA,
            xs: 1 << (log_n-1),
        };

        let params: Parameters =Parameters::new(&params_lit);
        let module = params.module();
    
        let log_q: usize = params.log_q();

        let mut source_xs: Source = Source::new(new_seed());

        let mut sk0: SecretKey = SecretKey::new(module);
        sk0.fill_ternary_hw(params.xs(), &mut source_xs);
        let mut sk0_ppol: base2k::SvpPPol = module.new_svp_ppol();
        module.svp_prepare(&mut sk0_ppol, &sk0.0);

        let mut sk1: SecretKey = SecretKey::new(module);
        sk1.fill_ternary_hw(params.xs(), &mut source_xs);
        let mut sk1_ppol: base2k::SvpPPol = module.new_svp_ppol();
        module.svp_prepare(&mut sk1_ppol, &sk1.0);

        let tmp_bytes: Vec<u8> = alloc_aligned(params.decrypt_rlwe_tmp_byte(log_q)| params.encrypt_rlwe_sk_tmp_bytes(log_q));

        Context{
            params: params,
            sk0: sk0,
            sk0_ppol: sk0_ppol,
            sk1: sk1,
            sk1_ppol: sk1_ppol,
            tmp_bytes: tmp_bytes,

        }
    }

    pub fn encrypt_rlwe_sk0(&mut self, pt: &Plaintext, ct: &mut Ciphertext<VecZnx>){

        let mut source_xe: Source = Source::new(new_seed());
        let mut source_xa: Source = Source::new(new_seed());

        encrypt_rlwe_sk(
            self.params.module(),
            ct.elem_mut(),
            Some(pt.elem()),
            &self.sk0_ppol,
            &mut source_xa,
            &mut source_xe,
            self.params.xe(),
            &mut self.tmp_bytes,
        );
    }

    pub fn encrypt_rlwe_sk1(&mut self, ct: &mut Ciphertext<VecZnx>, pt: &Plaintext){

        let mut source_xe: Source = Source::new(new_seed());
        let mut source_xa: Source = Source::new(new_seed());

        encrypt_rlwe_sk(
            self.params.module(),
            ct.elem_mut(),
            Some(pt.elem()),
            &self.sk1_ppol,
            &mut source_xa,
            &mut source_xe,
            self.params.xe(),
            &mut self.tmp_bytes,
        );
    }

    pub fn decrypt_sk0(&mut self, pt: &mut Plaintext, ct: &Ciphertext<VecZnx>){
        decrypt_rlwe(
            self.params.module(),
            pt.elem_mut(),
            ct.elem(),
            &self.sk0_ppol,
            &mut self.tmp_bytes,
        );
    }

    pub fn decrypt_sk1(&mut self, pt: &mut Plaintext, ct: &Ciphertext<VecZnx>){
        decrypt_rlwe(
            self.params.module(),
            pt.elem_mut(),
            ct.elem(),
            &self.sk1_ppol,
            &mut self.tmp_bytes,
        );
    }
}