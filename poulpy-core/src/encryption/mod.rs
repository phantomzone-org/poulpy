mod compressed;
mod gglwe_atk;
mod gglwe_ct;
mod gglwe_ksk;
mod gglwe_tsk;
mod ggsw_ct;
mod glwe_ct;
mod glwe_pk;
mod glwe_to_lwe_ksk;
mod lwe_ct;
mod lwe_ksk;
mod lwe_to_glwe_ksk;

pub const SIGMA: f64 = 3.2;
pub(crate) const SIGMA_BOUND: f64 = 6.0 * SIGMA;
