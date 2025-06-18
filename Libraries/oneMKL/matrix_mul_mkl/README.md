# `Matrix Multiplication with oneMKL` Sample

Matrix Multiplication with Intel® oneAPI Math Kernel Library (oneMKL) shows how to use the oneMKL optimized matrix multiplication routines, and provides a simple benchmark.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04 <br> Windows* 10, 11
| Hardware            | Skylake with Gen9 or newer
| Software            | Intel® oneAPI Math Kernel Library (oneMKL)
| What you will learn | How to use the oneMKL matrix multiplication functionality
| Time to complete    | 15 minutes

For more information on oneMKL and complete documentation of all oneMKL routines, see https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-documentation.html.

## Purpose

Matrix Multiplication uses oneMKL to multiply two large matrices and measure device performance.

This sample performs its computations on the default SYCL* device. You can set
the `ONEAPI_DEVICE_SELECTOR` environment variable to `"*:cpu"` or `"*:gpu"`
to select the device to use.
To find more information about the variable follow the link:
[ONEAPI_DEVICE_SELECTOR](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#oneapi_device_selector).

## Key Implementation Details

The oneMKL `blas::gemm` routine performs a matrix multiplication operation with optional scaling and updating behavior. oneMKL BLAS routines support both row-major and column-major matrix layouts; this sample uses the default column-major layout, the traditional choice for BLAS.

This sample provides a simple benchmark to test `gemm` performance on a SYCL device, and illustrates several best practices:
 - Perform a warmup run before timing, to allow oneMKL to initialize and prepare GEMM kernels for execution.
 - Pad matrix dimensions if needed to ensure data is well-aligned.

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

## Building the Matrix Multiplication with oneMKL Sample
> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
> Windows: C:\"Program Files (x86)"\Intel\oneAPI\setvars.bat
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### On a Linux* System
Run `make` to build and run the sample.

You can remove all generated files with `make clean`.

### On a Windows* System
Run `nmake` to build and run the sample. `nmake clean` removes temporary files.

## Running the Matrix Multiplication with oneMKL Sample

### Example of Output
Example output from this sample:

```
./matrix_mul_mkl single
oneMKL DPC++ GEMM benchmark
---------------------------
Device:                  Intel(R) Iris(R) Pro Graphics 580
Core/EU count:           72
Maximum clock frequency: 950 MHz

Benchmarking (4096 x 4096) x (4096 x 4096) matrix multiplication, single precision
 -> Initializing data...
 -> Warmup...
 -> Timing...

Average performance: ...

./matrix_mul_mkl double
oneMKL DPC++ GEMM benchmark
---------------------------
Device:                  Intel(R) Iris(R) Pro Graphics 580
Core/EU count:           72
Maximum clock frequency: 950 MHz

Benchmarking (4096 x 4096) x (4096 x 4096) matrix multiplication, double precision
 -> Initializing data...
 -> Warmup...
 -> Timing...

Average performance: ...
```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/current/overview.html).

## License

Code samples are licensed under the MIT license. See
[License.txt](License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](third-party-programs.txt).
