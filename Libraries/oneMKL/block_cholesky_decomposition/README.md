# `Block Cholesky Decomposition` Sample

Block Cholesky Decomposition shows how to use the oneMKL library's BLAS and LAPACK functionality to solve a symmetric, positive-definite block tridiagonal linear equation.

For more information on oneMKL, and complete documentation of all oneMKL routines, see https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04; Windows 10
| Hardware                          | Skylake with Gen9 or newer
| Software                          | Intel&reg; oneMKL
| What you will learn               | How to use oneMKL BLAS and LAPACK routines with pointer-based (USM) programming
| Time to complete                  | 15 minutes


## Purpose

Block Cholesky Decomposition consists of two small applications (`factor.cpp` and `solve.cpp`). The factor step generates a block tridiagonal matrix, then performs a block Cholesky factorization using oneMKL BLAS and LAPACK routines. The solver application uses this factorization to solve a linear system with the block tridiagonal matrix on the left-hand side. Both factoring and solving require several oneMKL routines. Some steps can be parallelized, while others must be ordered sequentially. The sample code shows how to inform oneMKL of the dependencies between routines using DPC++ events. The code uses pointer-based programming, with Unified Shared Memory (USM) throughout, which allows individual oneMKL routines to work on submatrices of the original matrices.

This sample will use the default DPC++ device. You can set the `SYCL_DEVICE_TYPE` environment variable to `cpu` or `gpu` to select the device to use.


## Key Implementation Details

This sample illustrates several important oneMKL routines: matrix multiplication, rank-k updates, and triangular solves from BLAS (`gemm`, `syrk`, and `trsm`), and Cholesky factorization (`potrf`) from LAPACK.


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Using Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

## Building the Block Cholesky Decomposition Sample
> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
Run `make` to build and run the factor and solve programs. You can remove all generated files with `make clean`.

### On a Windows* System
Run `nmake` to build and run the sample. `nmake clean` removes temporary files.

*Warning*: On Windows, static linking with oneMKL currently takes a very long time due to a known compiler issue. This will be addressed in an upcoming release.

## Running the Block Cholesky Decomposition Sample

### Example of Output
After building, if everything is working correctly, you will see the step-by-step output from the `factor` and `solve` programs. Each includes an accuracy check at the end to verify that the computation was successful.
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
Testing accuracy of the solution of linear equations system
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

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)