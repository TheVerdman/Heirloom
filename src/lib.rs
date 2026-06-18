#![forbid(unsafe_code)]

pub mod amp;
pub mod checkpoint;
pub mod data;
mod dispatch;
mod error;
pub mod extension;
mod grad_mode;
pub mod memory_transformer;
pub mod nn;
pub mod npy;
pub mod rng;
mod shape;
mod storage;
mod tensor;
pub mod tokenizer;

pub use dispatch::{
    operator_catalog, AliasPolicy, AutogradPolicy, DTypeRule, Operator, OperatorInfo, OperatorKind,
};
pub use error::{Result, TensorError};
pub use extension::{
    CustomUnaryBackward, CustomUnaryBackwardContext, CustomUnaryForward, CustomUnaryForwardContext,
    CustomUnaryOp, CustomUnaryRegistry,
};
pub use grad_mode::{is_grad_enabled, no_grad};
pub use storage::{DType, Device};
pub use tensor::Tensor;
