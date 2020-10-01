# Matrix Multiplication with oneMKL Sample

Matrix Multiplication with oneMKL shows how to use the oneMKL's optimized matrix multiplication routines. 

For more information on oneMKL, and complete documentation of all oneMKL routines, see https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04; Windows 10
| Hardware            | Skylake with Gen9 or newer
| Software            | Intel&reg; oneMKL beta
| What you will learn | How to use oneMKL's matrix multiplication functionality
| Time to complete    | 15 minutes


## Purpose

Matrix Multiplication uses oneMKL to multiply two large matrices.

This sample performs its computations on the default DPC++ device. You can set the `SYCL_DEVICE_TYPE` environment variable to `cpu` or `gpu` to select the device to use.


## Key Implementation Details

oneMKL's `blas::gemm` routine performs a generalized matrix multiplication operation. Both row-major and column-major matrix layouts are supported by oneMKL BLAS routines; this sample uses row-major layouts, the traditional choice for C++.

## License

This code sample is licensed under the MIT license.


## Building the Matrix Multiplication with oneMKL Sample

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
Run `make` to build and run the sample.

You can remove all generated files with `make clean`.

### On a Windows* System
Run `nmake` to build and run the sample. `nmake clean` removes temporary files.

## Running the Matrix Multiplication with oneMKL Sample

### Example of Output
If everything is working correctly, the program will generate two input matrices, and call oneMKL to multiply them. It will also compute the product matrix itself to verify the results from oneMKL.

```
./matrix_mul_mkl
Device: Intel(R) Gen9 HD Graphics NEO
Problem size:  A (600x1200) * B (1200x2400)  -->  C (600x2400)
Launching oneMKL GEMM calculation...
Performing reference calculation...
Results are accurate.
```
