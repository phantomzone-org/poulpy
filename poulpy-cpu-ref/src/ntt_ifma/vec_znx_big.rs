//! Large-coefficient (i128) ring element vector operations for [`NTTIfmaRef`](crate::NTTIfmaRef).
//!
//! `VecZnxBig` stores ring element vectors using `ScalarBig = i128` (one i128 per coefficient),
//! enabling exact CRT accumulation of NTT-domain products before normalization back to the
//! base-2^k representation.
//!
//! The i128 domain operations are backend-independent and are handled by the shared
//! `NTTIfmaVecZnxBigDefaults` in `hal_defaults`. This module only provides the
//! `I128BigOps` and `I128NormalizeOps` marker trait impls required by those defaults.

use crate::NTTIfmaRef;
use crate::reference::ntt120::{I128BigOps, I128NormalizeOps};

impl I128BigOps for NTTIfmaRef {}
impl I128NormalizeOps for NTTIfmaRef {}
