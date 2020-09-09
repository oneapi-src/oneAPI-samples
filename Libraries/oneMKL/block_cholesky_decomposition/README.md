# Block Cholesky Decomposition Sample

Block Cholesky Decomposition shows how to use the oneMKL library's BLAS and LAPACK functionality to solve a symmetric, positive-definite block tridiagonal linear equation.

For more information on oneMKL, and complete documentation of all oneMKL routines, see https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04; Windows 10
| Hardware                          | Skylake with Gen9 or newer
| Software                          | Intel&reg; oneMKL beta
| What you will learn               | How to use oneMKL BLAS and LAPACK routines with pointer-based (USM) programming
| Time to complete                  | 15 minutes


## Purpose

Block Cholesky Decomposition consists of two small applications (`factor.cpp` and `solve.cpp`). The factor step generates a block tridiagonal matrix, then performs a block Cholesky factorization using oneMKL BLAS and LAPACK routines. The solver application uses this factorization to solve a linear system with the block tridiagonal matrix on the left-hand side. Both factoring and solving require several oneMKL routines. Some steps can be parallelized, while others must be ordered sequentially. The sample code shows how to inform oneMKL of the dependencies that exist between routines, using DPC++ events. The code uses pointer-based programming, with Unified Shared Memory (USM), throughout, which allows individual oneMKL routines to work on submatrices of the original matrices.

This sample will use the default DPC++ device. You can set the `SYCL_DEVICE_TYPE` environment variable to `cpu` or `gpu` to select the device to use.


## Key Implementation Details

This sample illustrates several important oneMKL routines: matrix multiplication, rank-k updates, and triangular solves from BLAS (`gemm`, `syrk`, and `trsm`), and Cholesky factorization (`potrf`) from LAPACK.


## License

This code sample is licensed under the MIT license.


## Building the Block Cholesky Decomposition Sample

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
Run `make` to build and run the factor and solve programs. You can remove all generated files with `make clean`.

### On a Windows* System
Run `nmake` to build and run the sample. `nmake clean` removes temporary files.

## Running the Block Cholesky Decomposition Sample

### Example of Output
If everything is working correctly, after building you will see step-by-step output from the `factor` and `solve` programs. Each includes an accuracy check at the end to verify that the computation was successful.
```
./factor
Testing accuracy of Cholesky factorization
of randomly generated positive definite symmetric
block tridiagonal matrix by calculating residual.

Matrix size = 200
Block  size = 20
...
Matrices are being generated.
...
Call Cholesky factorization
...
Cholesky factorization succeeded.
Testing the residual
...
Residual test
||A-L*L^t||_F/||A||_F <= 5*EPS...
passed

./solve
Testing accuracy of solution of linear equations system
with randomly generated positive definite symmetric
block tridiagonal coefficient matrix by calculating
ratios of residuals to RHS vectors' norms.
...
Matrices are being generated.
...
Call Cholesky factorization
...
Cholesky factorization succeeded.
Call solving the system of linear equations
...
Solution succeeded.
The system is solved. Testing the residual
...
Residual test
max_(i=1,...,NRHS){||A*X(i)-F(i)||/||F(i)||} <= 10*EPS
passed
```
