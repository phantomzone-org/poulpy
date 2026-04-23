//! CKKS composite operations built from the primitive operation modules.
//!
//! ## Operations
//!
//! | Module | Operation |
//! |--------|-----------|
//! | [`add_many`] | Sum of many ciphertexts with deferred normalization |
//! | [`mul_many`] | Product of many ciphertexts through a balanced multiplication tree |
//! | [`mul_add`] | Fused multiply-add variants (`dst += a · b`) for product accumulation with scratch sizing and CKKS alignment handled by the API |
//! | [`mul_sub`] | Fused multiply-sub variants (`dst -= a · b`) for residual/update steps with scratch sizing and CKKS alignment handled by the API |
//! | [`dot_product`] | Inner products over ciphertext, plaintext-vector, and constant weights |

pub mod add_many;
pub mod dot_product;
pub mod mul_add;
pub mod mul_many;
pub mod mul_sub;

pub use add_many::CKKSAddManyOps;
pub use dot_product::CKKSDotProductOps;
pub use mul_add::CKKSMulAddOps;
pub use mul_many::CKKSMulManyOps;
pub use mul_sub::CKKSMulSubOps;
