# `cuRAND to oneMKL RNG Migration` Sample

The `cuRAND to oneMKL RNG Migration` Sample is a collection of code samples that demonstrate the cuRAND equivalent SYCL API functionality in the Intel® oneAPI Math Kernel Library (oneMKL). 

| Area                   | Description
|:---                    |:---
| What you will learn    | How to migrate cuRAND API based source code to the equivalent SYCL*-compliant oneMKL Interfaces API for random number generation (RNG)
| Time to complete       | 30 minutes
| Category               | Code Optimization

For more information on oneMKL and complete documentation of all oneMKL routines, see https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-documentation.html.

## Purpose

The sample source code using SYCL was migrated from CUDA source code for offloading computations to a GPU/CPU. The sample demonstrates migrating code to SYCL, optimizing the migration steps, and improving execution time.

Each cuRAND sample source file shows the usage of different oneMKL RNG domain routines. All are basic programs demonstrating the usage for a single method of generating pseudorandom numbers.

>**Note**: This sample is based on the [*cuRAND Library - APIs Examples*](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuRAND) samples in the NVIDIA/CUDALibrarySamples GitHub repository.

## Prerequisites

| Optimized for         | Description
|:---                   |:---
| OS                    | Ubuntu* 20.04
| Hardware              | 10th Gen Intel® processors or newer
| Software              | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details

This sample contains two sets of sources in the following folders:

| Folder Name             | Description
|:---                     |:---
| `01_sycl_dpct_output`   | Contains initial output of the Intel® DPC++ Compatibility Tool used to migrate SYCL-compliant code from CUDA code. <br> It may contain not fully migrated or incorrectly generated code that has to be manually fixed before it is functional. (The code does not work as supplied.)
| `02_sycl_dpct_migrated` | Contains CUDA to SYCL migrated code generated using the Intel® DPC++ Compatibility Tool with the manual changes implemented to make the code fully functional.

These functions are classified into eight different directories, each based on an RNG engine. There are **48** samples:

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `cuRAND Migration` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*.

### On Linux*

1. Change to the sample directory.
2. Build the samples.
   ```
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make
   ```

   By default, this command sequence builds the version of the source code in the  `02_sycl_dpct_migrated` folder.

## Run the `cuRAND Migration` Sample

### On Linux

Run the programs on a CPU or GPU. Each sample uses a default device, which in most cases is a GPU.

1. Run the samples in the `02_sycl_dpct_migrated` folder.
   ```
   make run_mt19937_uniform
   ```

## Example Output

This is example output if you built the default and ran `run_mt19937_uniform`.

```
Scanning dependencies of target mt19937_uniform
[ 50%] Building CXX object 02_sycl_dpct_migrated/mt19937/CMakeFiles/mt19937_uniform.dir/mt19937_uniform.cpp.o
[100%] Linking CXX executable ../../bin/mt19937_uniform
[100%] Built target mt19937_uniform
Host
0.966454
0.778166
0.440733
0.116851
0.007491
0.090644
0.910976
0.942535
0.939269
0.807002
0.582228
0.034926
=====
Device
0.966454
0.778166
0.440733
0.116851
0.007491
0.090644
0.910976
0.942535
0.939269
0.807002
0.582228
0.034926
=====
[100%] Built target run_mt19937_uniform
```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/current/overview.html).

## License

Code samples are licensed under the MIT license. See
[License.txt](License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](third-party-programs.txt).
