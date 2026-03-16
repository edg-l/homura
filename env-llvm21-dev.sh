#!/bin/bash
# Source this to build homura against the local patched LLVM 21.
# Usage: source env-llvm21-dev.sh && cargo test ...
export LLVM_SYS_211_PREFIX=/home/edgar/data/llvm-21
export MLIR_SYS_210_PREFIX=/home/edgar/data/llvm-21
export MLIR_SYS_LINK_SHARED=1
export RUSTFLAGS="-C link-args=-Wl,-rpath,/home/edgar/data/llvm-21/lib"
echo "Using patched LLVM 21 from /home/edgar/data/llvm-21"
