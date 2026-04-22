//! CKKS composite operations built from the primitive operation modules.

pub mod add_many;
pub mod dot_product;
pub mod mul_add;
pub mod mul_many;

pub use add_many::CKKSAddManyOps;
pub use dot_product::CKKSDotProductOps;
pub use mul_add::CKKSMulAddOps;
pub use mul_many::CKKSMulManyOps;
