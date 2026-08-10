import os

def test_rust_ext_cargo_toml_exists():
    assert os.path.exists("rust_ext/Cargo.toml"), "Rust microkernel Cargo.toml is missing."

def test_rust_deterministic_primitives():
    # Verify the structure has the required math modules
    assert os.path.exists("rust_ext/src/risk.rs")
    assert os.path.exists("rust_ext/src/statistics.rs")
    assert os.path.exists("rust_ext/src/portfolio.rs")
