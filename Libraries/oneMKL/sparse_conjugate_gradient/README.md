# `Sparse Conjugate Gradient` Sample
Sparse Conjugate Gradient shows how to use the Intel® oneAPI Math Kernel Library (oneMKL) sparse linear algebra functionality to solve a sparse, symmetric linear system using the (preconditioned) conjugate gradient method.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04 <br> Windows* 10, 11
| Hardware            | Skylake with Gen9 or newer
| Software            | Intel® oneAPI Math Kernel Library (oneMKL)
| What you will learn | How to use oneMKL sparse linear algebra functionality
| Time to complete    | 15 minutes

For more information on oneMKL and complete documentation of all oneMKL routines, see https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-documentation.html.

## Purpose
Sparse Conjugate Gradient uses oneMKL sparse linear algebra routines to solve a system of linear equations Ax = b, where the A matrix is symmetric and sparse. The symmetric Gauss-Seidel preconditioner is used to accelerate convergence.

This sample performs its computations on the default SYCL* device. You can set the `SYCL_DEVICE_TYPE` environment variable to `cpu` or `gpu` to select the device to use.

## Key Implementation Details
oneMKL sparse routines use a two-stage method where the sparse matrix is analyzed to prepare subsequent calculations (the _optimize_ step). Sparse matrix-vector multiplication and triangular solves (`gemv` and `trsv`) are used to implement the main loop, along with vector routines from BLAS. Two implementations are provided: The first implementation, in `sparse_cg.cpp`, has several places where a device to host copy and wait are initiated to allow the alpha and beta coefficients to be initiated in the BLAS vector routines as host scalars.  The second implementation, in `sparse_cg2.cpp`, keeps the coefficients for alpha and beta on the device, which require that custom axpby2 and axpy3 functions are written to handle the construction of alpha and beta coefficients on-the-fly from the device. This removes some of the synchronization points that are seen in the first implementation.

## Using Visual Studio Code* (Optional)
You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel Software Developer Tools**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel Software Developer Tools**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see the
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

## Building the Sparse Conjugate Gradient Sample
> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `$ bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\"Program Files (x86)"\Intel\oneAPI\setvars.bat`
> - For Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or MacOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### On a Linux* System
Run `make` to build and run the sample.

You can remove all generated files with `make clean.`

### On a Windows* System
Run `nmake` to build and run the sample. `nmake clean` removes temporary files.

> **Warning**: On Windows, static linking with oneMKL currently takes a very long time due to a known compiler issue. This will be addressed in an upcoming release.

## Running the Sparse Conjugate Gradient Sample

### Example of Output
If everything is working correctly, the example programs will rapidly converge to a solution. Each test will run in both single and double precision (if available on the selected device).

The first PCG implementation with host side coefficients:
```
./sparse_cg
########################################################################
# Sparse Preconditioned Conjugate Gradient Solver with USM
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
# alpha and beta constants in PCG algorithm are host side.
#
########################################################################

Running tests on Intel(R) Data Center GPU Max 1550.
        Running with single precision real data type:

                sparse PCG parameters:
                        A size: (4096, 4096)
                        Preconditioner = Symmetric Gauss-Seidel
                        max iterations = 500
                        relative tolerance limit = 1e-05
                        absolute tolerance limit = 0.0005
                                relative norm of residual on    1 iteration: 0.178532
                                relative norm of residual on    2 iteration: 0.0280123
                                relative norm of residual on    3 iteration: 0.0048948
                                relative norm of residual on    4 iteration: 0.000796108
                                relative norm of residual on    5 iteration: 0.000119025
                                relative norm of residual on    6 iteration: 1.86945e-05
                                absolute norm of residual on    6 iteration: 0.000149556

                Preconditioned CG process has successfully converged in absolute error in    6 steps with
                 relative error ||r||_2 / ||r_0||_2 = 1.86945e-05 > 1e-05
                 absolute error ||r||_2             = 0.000149556 < 0.0005

        Running with double precision real data type:

                sparse PCG parameters:
                        A size: (4096, 4096)
                        Preconditioner = Symmetric Gauss-Seidel
                        max iterations = 500
                        relative tolerance limit = 1e-05
                        absolute tolerance limit = 0.0005
                                relative norm of residual on    1 iteration: 0.178532
                                relative norm of residual on    2 iteration: 0.0280123
                                relative norm of residual on    3 iteration: 0.0048948
                                relative norm of residual on    4 iteration: 0.000796108
                                relative norm of residual on    5 iteration: 0.000119025
                                relative norm of residual on    6 iteration: 1.86945e-05
                                absolute norm of residual on    6 iteration: 0.000149556

                Preconditioned CG process has successfully converged in absolute error in    6 steps with
                 relative error ||r||_2 / ||r_0||_2 = 1.86945e-05 > 1e-05
                 absolute error ||r||_2             = 0.000149556 < 0.0005

```

and the second PCG implementation with device side coefficients:
```
./sparse_cg2
########################################################################
# Sparse Preconditioned Conjugate Gradient Solver with USM 2
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
# alpha and beta constants in PCG algorithm are kept
# device side.
#
########################################################################

Running tests on Intel(R) Data Center GPU Max 1550.
        Running with single precision real data type:

                sparse PCG parameters:
                        A size: (4096, 4096)
                        Preconditioner = Symmetric Gauss-Seidel
                        max iterations = 500
                        relative tolerance limit = 1e-05
                        absolute tolerance limit = 0.0005
                                relative norm of residual on    1 iteration: 0.178532
                                relative norm of residual on    2 iteration: 0.0280123
                                relative norm of residual on    3 iteration: 0.0048948
                                relative norm of residual on    4 iteration: 0.000796109
                                relative norm of residual on    5 iteration: 0.000119025
                                relative norm of residual on    6 iteration: 1.86945e-05
                                absolute norm of residual on    6 iteration: 0.000149556

                Preconditioned CG process has successfully converged in absolute error in    6 steps with
                 relative error ||r||_2 / ||r_0||_2 = 1.86945e-05 > 1e-05
                 absolute error ||r||_2             = 0.000149556 < 0.0005

        Running with double precision real data type:

                sparse PCG parameters:
                        A size: (4096, 4096)
                        Preconditioner = Symmetric Gauss-Seidel
                        max iterations = 500
                        relative tolerance limit = 1e-05
                        absolute tolerance limit = 0.0005
                                relative norm of residual on    1 iteration: 0.178532
                                relative norm of residual on    2 iteration: 0.0280123
                                relative norm of residual on    3 iteration: 0.0048948
                                relative norm of residual on    4 iteration: 0.000796108
                                relative norm of residual on    5 iteration: 0.000119025
                                relative norm of residual on    6 iteration: 1.86945e-05
                                absolute norm of residual on    6 iteration: 0.000149556

                Preconditioned CG process has successfully converged in absolute error in    6 steps with
                 relative error ||r||_2 / ||r_0||_2 = 1.86945e-05 > 1e-05
                 absolute error ||r||_2             = 0.000149556 < 0.0005

```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/current/overview.html).

## License

Code samples are licensed under the MIT license. See
[License.txt](License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](third-party-programs.txt).
