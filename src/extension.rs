use crate::{DType, Device, Result, TensorError};
use std::fmt;

pub type CustomUnaryForward = for<'a> fn(CustomUnaryForwardContext<'a>) -> Result<Vec<f64>>;
pub type CustomUnaryBackward = for<'a> fn(CustomUnaryBackwardContext<'a>) -> Result<Vec<f64>>;

#[derive(Clone, Copy)]
pub struct CustomUnaryOp {
    name: &'static str,
    forward: CustomUnaryForward,
    backward: CustomUnaryBackward,
}

#[derive(Clone, Copy)]
pub struct CustomUnaryForwardContext<'a> {
    pub name: &'static str,
    pub input: &'a [f64],
    pub shape: &'a [usize],
    pub dtype: DType,
    pub device: Device,
}

#[derive(Clone, Copy)]
pub struct CustomUnaryBackwardContext<'a> {
    pub name: &'static str,
    pub input: &'a [f64],
    pub output: &'a [f64],
    pub grad_output: &'a [f64],
    pub shape: &'a [usize],
    pub dtype: DType,
    pub device: Device,
}

impl CustomUnaryOp {
    pub fn new(
        name: &'static str,
        forward: CustomUnaryForward,
        backward: CustomUnaryBackward,
    ) -> Result<Self> {
        validate_custom_op_name(name)?;
        Ok(Self {
            name,
            forward,
            backward,
        })
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub(crate) fn forward(
        &self,
        input: &[f64],
        shape: &[usize],
        dtype: DType,
        device: Device,
    ) -> Result<Vec<f64>> {
        (self.forward)(CustomUnaryForwardContext {
            name: self.name,
            input,
            shape,
            dtype,
            device,
        })
    }

    pub(crate) fn backward(
        &self,
        input: &[f64],
        output: &[f64],
        grad_output: &[f64],
        shape: &[usize],
        dtype: DType,
        device: Device,
    ) -> Result<Vec<f64>> {
        (self.backward)(CustomUnaryBackwardContext {
            name: self.name,
            input,
            output,
            grad_output,
            shape,
            dtype,
            device,
        })
    }
}

impl fmt::Debug for CustomUnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomUnaryOp")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for CustomUnaryForwardContext<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomUnaryForwardContext")
            .field("name", &self.name)
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for CustomUnaryBackwardContext<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomUnaryBackwardContext")
            .field("name", &self.name)
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Default)]
pub struct CustomUnaryRegistry {
    ops: Vec<CustomUnaryOp>,
}

impl CustomUnaryRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, op: CustomUnaryOp) -> Result<()> {
        if self.ops.iter().any(|existing| existing.name() == op.name()) {
            return Err(TensorError::InvalidOperation(format!(
                "custom unary op {:?} is already registered",
                op.name()
            )));
        }
        self.ops.push(op);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<CustomUnaryOp> {
        self.ops.iter().copied().find(|op| op.name() == name)
    }

    pub fn operators(&self) -> &[CustomUnaryOp] {
        &self.ops
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

fn validate_custom_op_name(name: &str) -> Result<()> {
    if name.trim().is_empty() {
        return Err(TensorError::InvalidOperation(
            "custom op name cannot be empty".to_string(),
        ));
    }
    if name.len() > 128 {
        return Err(TensorError::InvalidOperation(format!(
            "custom op name must be at most 128 bytes, got {}",
            name.len()
        )));
    }
    if !name
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | ':' | '-'))
    {
        return Err(TensorError::InvalidOperation(format!(
            "custom op name {name:?} contains unsupported characters"
        )));
    }
    Ok(())
}
