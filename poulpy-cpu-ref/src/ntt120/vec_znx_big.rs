//! Large-coefficient (i128) ring element vector support for [`NTT120Ref`](crate::NTT120Ref).
//!
//! The shared `poulpy-hal` NTT120 defaults rely on backend-provided `I128BigOps`
//! and `I128NormalizeOps` hooks for vectorized i128 operations.

use crate::NTT120Ref;
use crate::reference::ntt120::{I128BigOps, I128NormalizeOps};

impl I128BigOps for NTT120Ref {}
impl I128NormalizeOps for NTT120Ref {}
