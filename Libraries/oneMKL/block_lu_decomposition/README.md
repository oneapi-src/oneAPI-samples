# `Block LU Decomposition` Sample

Block LU Decomposition shows how to use the oneMKL library's BLAS and LAPACK functionality to solve a block tridiagonal linear equation.

For more information on oneMKL, and complete documentation of all oneMKL routines, see https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04; Windows 10
| Hardware                          | Skylake with Gen9 or newer
| Software                          | Intel&reg; oneMKL
| What you will learn               | How to use oneMKL BLAS and LAPACK routines with pointer-based (USM) programming
| Time to complete                  | 15 minutes


## Purpose

Block LU Decomposition consists of two small applications (`factor.cpp` and `solve.cpp`). 
The factor.cpp generates a tridiagonal block matrix, then performs a block LU factorization using oneMKL BLAS and LAPACK routines. The solve.cpp application uses this factorization to solve a linear system with the block tridiagonal matrix on the left-hand side. 
Both factoring and solving require several oneMKL routines. Some steps can be parallelized, while others must be ordered sequentially. The sample code shows how to inform oneMKL of the existing dependencies between routines using DPC++ events. This code sample uses pointer-based programming with Unified Shared Memory (USM), allowing individual oneMKL routines to work on submatrices of the original matrices.

This sample will use the default DPC++ device. You can set the `SYCL_DEVICE_TYPE` environment variable to `cpu` or `gpu` to select the device to use.


## Key Implementation Details

This sample illustrates several important oneMKL routines: matrix multiplication, triangular solves from BLAS (`gemm`, `trsm`), and LU factorization (`getrf`) from LAPACK, as well as several other utility routines.


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)


## Building the Block LU Decomposition Sample

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
Run `make` to build and run the factor and solve programs. You can remove all generated files with `make clean`.

### On a Windows* System
Run `nmake` to build and run the sample. `nmake clean` removes temporary files.

*Warning*: On Windows, static linking with oneMKL currently takes a very long time due to a known compiler issue. This will be addressed in an upcoming release.

## Running the Block LU Decomposition Sample

### Example of Output
After building, if everything is working correctly, you will see the output from both the `factor` and `solve` programs. Each includes an accuracy check at the end. The example output below shows a successful run with a very small floating-point error.
```
./factor
Testing the accuracy of LU factorization with pivoting
of randomly generated block tridiagonal matrix
by calculating norm of the residual matrix.
||A - LU||_F/||A||_F = 3.65246e-16

./solve
Testing accuracy of solution of linear equations system
with randomly generated block tridiagonal coefficient
matrix by calculating ratios of residuals
to RHS vectors norms.
max_(i=1,...,nrhs){||ax(i)-f(i)||/||f(i)||} = 6.88457e-13
```
