# Productive libraries

This directory includes OneMKL-compatible implementations of BLAS kernels that are written in a productive language for FPGAs. The kernels are then compiled into SYCL code.

## File structure

* [blas](blas/README.md) - BLAS kernels.
* `include`: common headers used by the kernels.
* `test`: Google test infrastructure and common headers for tests
* `tools` - compiler and headers of the productive language
