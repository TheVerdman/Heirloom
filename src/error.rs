use std::error::Error;
use std::fmt;

pub type Result<T> = std::result::Result<T, TensorError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    Shape(String),
    DType(String),
    Device(String),
    Io(String),
    InvalidOperation(String),
    Autograd(String),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(message) => write!(f, "shape error: {message}"),
            Self::DType(message) => write!(f, "dtype error: {message}"),
            Self::Device(message) => write!(f, "device error: {message}"),
            Self::Io(message) => write!(f, "io error: {message}"),
            Self::InvalidOperation(message) => write!(f, "invalid operation: {message}"),
            Self::Autograd(message) => write!(f, "autograd error: {message}"),
        }
    }
}

impl Error for TensorError {}
