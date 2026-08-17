use pyo3::exceptions::{PyValueError, PyRuntimeError};
use pyo3::prelude::*;
use std::fmt;

#[derive(Debug)]
pub enum KernelError {
    MathError(String),
    InputError(String),
}

impl std::error::Error for KernelError {}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            KernelError::MathError(msg) => write!(f, "Math Error: {}", msg),
            KernelError::InputError(msg) => write!(f, "Input Error: {}", msg),
        }
    }
}

impl std::convert::From<KernelError> for PyErr {
    fn from(err: KernelError) -> PyErr {
        match err {
            KernelError::InputError(msg) => PyValueError::new_err(msg),
            KernelError::MathError(msg) => PyRuntimeError::new_err(msg),
        }
    }
}
