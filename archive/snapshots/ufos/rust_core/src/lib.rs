pub mod memory;
pub mod execution;
pub mod strategy;
pub mod mcp;
pub mod risk;
pub mod networking;
pub mod bridge;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn ufos_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<bridge::UfosBridge>()?;
    Ok(())
}
