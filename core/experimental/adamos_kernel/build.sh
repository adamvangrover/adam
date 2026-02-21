#!/bin/bash
# AdamOS Kernel Build Script (Rust -> WASM)
# -----------------------------------------------------------------------------
# Prerequisites:
# 1. Install Rust: https://rustup.rs/
# 2. Install wasm-pack: https://rustwasm.github.io/wasm-pack/installer/
#
# Usage:
# ./build.sh
# -----------------------------------------------------------------------------

set -e

echo "🚀 Building AdamOS Kernel (WASM)..."

if ! command -v wasm-pack &> /dev/null; then
    echo "❌ Error: wasm-pack is not installed."
    echo "Please install it via: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    exit 1
fi

# Build the WASM module
# Output will be in pkg/ directory
wasm-pack build --target web --out-dir ../../../showcase/js/adamos_pkg

echo "✅ Build Complete! WASM artifacts are in showcase/js/adamos_pkg/"
echo "The adamos_bridge.js will now attempt to load this module."
