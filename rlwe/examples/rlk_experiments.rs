use base2k::{
    Encoding, Infos, Module, Sampling, SvpPPol, SvpPPolOps, VecZnx, VecZnxDftOps, VecZnxOps,
    VmpPMat, VmpPMatOps, is_aligned,
};
use itertools::izip;
use rlwe::ciphertext::{Ciphertext, new_gadget_ciphertext};
use rlwe::elem::ElemCommon;
use rlwe::encryptor::encrypt_rlwe_sk;
use rlwe::keys::SecretKey;
use rlwe::plaintext::Plaintext;
use sampling::source::{Source, new_seed};

fn main() {
    let n: usize = 32;
    let module: Module = Module::new(n, base2k::MODULETYPE::FFT64);
    let log_base2k: usize = 16;
    let log_k: usize = 32;
    let cols: usize = 4;

    let mut a: VecZnx = module.new_vec_znx(cols);
    let mut data: Vec<i64> = vec![0i64; n];
    data[0] = 0;
    data[1] = 0;
    a.encode_vec_i64(log_base2k, log_k, &data, 16);

    let mut a_dft: base2k::VecZnxDft = module.new_vec_znx_dft(cols);

    module.vec_znx_dft(&mut a_dft, &a, cols);

    (0..cols).for_each(|i| {
        println!("{:?}", a_dft.at::<f64>(&module, i));
    })
}

pub struct GadgetCiphertextProtocol {}

impl GadgetCiphertextProtocol {
    pub fn new() -> GadgetCiphertextProtocol {
        Self {}
    }

    pub fn allocate(
        module: &Module,
        log_base2k: usize,
        rows: usize,
        log_q: usize,
    ) -> GadgetCiphertextShare {
        GadgetCiphertextShare::new(module, log_base2k, rows, log_q)
    }

    pub fn gen_share(
        module: &Module,
        sk: &SecretKey,
        pt: &Plaintext,
        seed: &[u8; 32],
        share: &mut GadgetCiphertextShare,
        tmp_bytes: &mut [u8],
    ) {
        share.seed.copy_from_slice(seed);
        let mut source_xe: Source = Source::new(new_seed());
        let mut source_xa: Source = Source::new(*seed);
        let mut sk_ppol: SvpPPol = module.new_svp_ppol();
        sk.prepare(module, &mut sk_ppol);
        share.value.iter_mut().for_each(|ai| {
            //let elem = Elem<VecZnx>{};
            //encrypt_rlwe_sk_thread_safe(module, ai, Some(pt.elem()), &sk_ppol, &mut source_xa, &mut source_xe, 3.2, tmp_bytes);
        })
    }
}

pub struct GadgetCiphertextShare {
    pub seed: [u8; 32],
    pub log_q: usize,
    pub log_base2k: usize,
    pub value: Vec<VecZnx>,
}

impl GadgetCiphertextShare {
    pub fn new(module: &Module, log_base2k: usize, rows: usize, log_q: usize) -> Self {
        let value: Vec<VecZnx> = Vec::new();
        let cols: usize = (log_q + log_base2k - 1) / log_base2k;
        (0..rows).for_each(|_| {
            let vec_znx: VecZnx = module.new_vec_znx(cols);
        });
        Self {
            seed: [u8::default(); 32],
            log_q: log_q,
            log_base2k: log_base2k,
            value: value,
        }
    }

    pub fn rows(&self) -> usize {
        self.value.len()
    }

    pub fn cols(&self) -> usize {
        self.value[0].cols()
    }

    pub fn aggregate_inplace(&mut self, module: &Module, a: &GadgetCiphertextShare) {
        izip!(self.value.iter_mut(), a.value.iter()).for_each(|(bi, ai)| {
            module.vec_znx_add_inplace(bi, ai);
        })
    }

    pub fn get(&self, module: &Module, b: &mut Ciphertext<VmpPMat>, tmp_bytes: &mut [u8]) {
        assert!(is_aligned(tmp_bytes.as_ptr()));

        let rows: usize = b.rows();
        let cols: usize = b.cols();

        assert!(tmp_bytes.len() >= gadget_ciphertext_share_get_tmp_bytes(module, rows, cols));

        assert_eq!(self.value.len(), rows);
        assert_eq!(self.value[0].cols(), cols);

        let (tmp_bytes_vmp_prepare_row, tmp_bytes_vec_znx) =
            tmp_bytes.split_at_mut(module.vmp_prepare_tmp_bytes(rows, cols));

        let mut c: VecZnx = VecZnx::from_bytes_borrow(module.n(), cols, tmp_bytes_vec_znx);

        let mut source: Source = Source::new(self.seed);

        (0..self.value.len()).for_each(|row_i| {
            module.vmp_prepare_row(
                b.at_mut(0),
                self.value[row_i].raw(),
                row_i,
                tmp_bytes_vmp_prepare_row,
            );
            module.fill_uniform(self.log_base2k, &mut c, cols, &mut source);
            module.vmp_prepare_row(b.at_mut(1), c.raw(), row_i, tmp_bytes_vmp_prepare_row)
        })
    }

    pub fn get_new(&self, module: &Module, tmp_bytes: &mut [u8]) -> Ciphertext<VmpPMat> {
        let mut b: Ciphertext<VmpPMat> =
            new_gadget_ciphertext(module, self.log_base2k, self.rows(), self.log_q);
        self.get(module, &mut b, tmp_bytes);
        b
    }
}

pub fn gadget_ciphertext_share_get_tmp_bytes(module: &Module, rows: usize, cols: usize) -> usize {
    module.vmp_prepare_tmp_bytes(rows, cols) + module.bytes_of_vec_znx(cols)
}

pub struct CircularCiphertextProtocol {}

pub struct CircularGadgetCiphertextProtocol {}
