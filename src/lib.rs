#![forbid(unsafe_code)]
#![warn(rustdoc::broken_intra_doc_links)]

//! A small, inspectable tensor and autograd runtime with explicit CPU and CUDA
//! execution paths.
//!
//! Heirloom's core review path is [`Tensor`] → reverse-mode autograd → checked
//! kernel dispatch → the modules in [`nn`] → [`memory_transformer`]. CUDA FFI
//! and PTX loading are isolated in the `heirloom-kernels` crate; this crate
//! forbids unsafe Rust.
//!
//! # Autograd example
//!
//! ```
//! use heirloom::{Result, Tensor};
//!
//! fn main() -> Result<()> {
//!     let x = Tensor::from_f32(vec![2.0, -3.0], &[2], true)?;
//!     let loss = x.mul(&x)?.sum()?;
//!     loss.backward()?;
//!
//!     assert_eq!(x.grad(), Some(vec![4.0, -6.0]));
//!     Ok(())
//! }
//! ```

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
