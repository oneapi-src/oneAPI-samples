# `Sparse Conjugate Gradient` Sample

Sparse Conjugate Gradient shows how to use the oneMKL library's sparse linear algebra functionality to solve a sparse, symmetric linear system using the (preconditioned) conjugate gradient method.

For more information on oneMKL, and complete documentation of all oneMKL routines, see https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04; Windows 10
| Hardware            | Skylake with Gen9 or newer
| Software            | Intel&reg; oneMKL
| What you will learn | How to use oneMKL's sparse linear algebra functionality
| Time to complete    | 15 minutes


## Purpose

Sparse Conjugate Gradient uses oneMKL's sparse linear algebra routines to solve a system of linear equations Ax = b, where the A matrix is symmetric and sparse. The symmetric Gauss-Seidel preconditioner is used to accelerate convergence.

This sample performs its computations on the default DPC++ device. You can set the `SYCL_DEVICE_TYPE` environment variable to `cpu` or `gpu` to select the device to use.


## Key Implementation Details

oneMKL sparse routines use a two-stage method where the sparse matrix is analyzed to prepare subsequent calculations (the _optimize_ step). Sparse matrix-vector multiplication and triangular solves (`gemv` and `trsv`) are used to implement the main loop, along with vector routines from BLAS.


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)


## Building the Sparse Conjugate Gradient Sample

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
Run `make` to build and run the sample.

You can remove all generated files with `make clean.`

### On a Windows* System
Run `nmake` to build and run the sample. `nmake clean` removes temporary files.

*Warning*: On Windows, static linking with oneMKL currently takes a very long time due to a known compiler issue. This will be addressed in an upcoming release.

## Running the Sparse Conjugate Gradient Sample

### Example of Output
If everything is working correctly, the example program will rapidly converge to a solution and display the solution vector's first few entries. The test will run in both single and double precision (if available on the selected device).
```
./sparse_cg
########################################################################
# Sparse Conjugate Gradient Solver
#
# Uses the preconditioned conjugate gradient algorithm to
# iteratively solve the symmetric linear system
#
#     A * x = b
#
# where A is a symmetric sparse matrix in CSR format, and
#       x and b are dense vectors.
#
# Uses the symmetric Gauss-Seidel preconditioner.
#
########################################################################

Running tests on Intel(R) Gen9 HD Graphics NEO.
        Running with single precision real data type:
                relative norm of residual on 1 iteration: 0.0856119
                relative norm of residual on 2 iteration: 0.00204826
                relative norm of residual on 3 iterations: 6.68015e-05

                Preconditioned CG process has successfully converged, and
                the following solution has been obtained:

                x[0] = 0.0666633
                x[1] = 0.0835483
                x[2] = 0.0835491
                x[3] = 0.0666627
                ...
        Running with double precision real data type:
                relative norm of residual on 1 iteration: 0.0856119
                relative norm of residual on 2 iteration: 0.00204827
                relative norm of residual on 3 iteration: 6.68017e-05

                Preconditioned CG process has successfully converged, and
                the following solution has been obtained:

                x[0] = 0.0666633
                x[1] = 0.0835483
                x[2] = 0.0835491
                x[3] = 0.0666627
                ...
```
