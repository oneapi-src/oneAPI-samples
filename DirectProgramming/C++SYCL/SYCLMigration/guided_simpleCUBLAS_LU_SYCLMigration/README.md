# `simpleCUBLAS_LU` Sample

LU decomposition factors a matrix as the product of an upper triangular matrix and lower triangular matrix. The original CUDA* source code is migrated to SYCL for portability across GPUs from multiple vendors.

| Area              | Description
|:---                   |:---
| What you will learn              | Migrate simpleCUBLAS_LU from CUDA to SYCL
| Time to complete              | 15 minutes
| Category                      | Concepts and Functionality

## Purpose

The sample shows the migration of simpleCUBLAS_LU from CUDA to SYCL using SYCLomatic tool and optimizing the migrated sycl code further to achieve good results.


>**Note**: We use Intel® open-sources SYCLomatic migration tool which assists developers in porting CUDA code automatically to SYCL code. To finish the process, developers complete the rest of the coding manually and then tune to the desired level of performance for the target architecture. Users can also use Intel® DPC++ Compatibility Tool which comes along with the Intel® oneAPI Base Toolkit.

This sample contains two versions in the following folders:

| Folder Name                   | Description
|:---                           |:---
| `01_dpct_output`              | Contains the output of the SYCLomatic tool used to migrate SYCL-compliant code from CUDA code. The tool completely migrates code but needs manual changes to get functional correctness on the given list of hardware.
| `02_sycl_migrated`            | Contains migrated SYCL code from CUDA code with manual changes.

## Prerequisites

| Optimized for              | Description
|:---                   |:---
| OS                    | Ubuntu* 22.04
| Hardware              | Intel® Gen9 <br> Intel® Gen11 <br> Intel® Xeon CPU <br> Intel® Data Center GPU Max <br> NVIDIA Tesla P100 <br> NVIDIA A100 <br> NVIDIA H100
| Software                | SYCLomatic (Tag - 20240116) <br> Intel® oneAPI Base Toolkit version 2024.0.0 <br> oneAPI for NVIDIA GPUs" plugin from Codeplay (version 2024.0.0)

For more information on how to install Syclomatic Tool & DPC++ CUDA® plugin, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.v354cy) <br> How to run SYCL™ applications on NVIDIA® GPUs, refer to oneAPI for NVIDIA GPUs plugin from Codeplay [Install oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/)

## Key Implementation Details

This sample demonstrates the migration of the following:

- CUBLAS Library, LU decomposition

>  **Note**: Refer to [Workflow for a CUDA* to SYCL* Migration](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html#gs.s2njvh) for general information about the migration workflow.

### CUDA source code evaluation

This example demonstrates how to use the cuBLAS library API for lower-upper (LU) decomposition of a matrix. This sample uses 10000 matrices of size 4x4 and performs LU decomposition of them using batched decomposition API of cuBLAS library. To test the correctness of upper and lower matrices generated, they are multiplied and compared with the original input matrix.

This sample is migrated from the NVIDIA CUDA sample. See the sample [simpleCUBLAS_LU](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/simpleCUBLAS_LU) in the NVIDIA/cuda-samples GitHub.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Migrate the `simpleCUBLAS_LU` Sample

### Migrate the Code using SYCLomatic

For this sample, the SYCLomatic tool automatically migrates 100% of the CUDA runtime API's to SYCL. Follow these steps to generate the SYCL code using the compatibility tool:

1. Clone the required GitHub repository to your local environment.
   ```
   git clone https://github.com/NVIDIA/cuda-samples.git
   ```
2. Change to the simpleCUBLAS_LU sample directory.
   ```
   cd cuda-samples/Samples/4_CUDA_Libraries/simpleCUBLAS_LU
   ```
3. Generate a compilation database with intercept-build
   ```
   intercept-build make
   ```
   The above step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.

4. Pass the JSON file as input to the Intel® SYCLomatic Compatibility Tool. The result is written to a folder named dpct_output. The --in-root specifies path to the root of the source tree to be migrated. The --gen-helper-function option will make a copy of dpct header files/functions used in the migrated code into the dpct_output folder as include folder.
In the native CUDA code, there are two cublas APIs that cannot be enabled (defined by MACROs) during one compilation. So, in one c2s execution, only one code path can be migrated. To get both the APIs migrated we need to exclude the line #define DOUBLE_PRECISION in native cuda code and execute c2s twice as shown below.

   ```
   c2s --in-root ../../.. --gen-helper-function --out-root out -p . --extra-arg="-DDOUBLE_PRECISION"
   c2s --in-root ../../../ --out-root out -p .
   ```

### Manual Workaround
CUDA code includes a custom API findCUDADevice in helper_cuda file to find the best CUDA Device available
```
 findCudaDevice (argc, (const char **) argv);   
```
Since its a custom API SYCLomatic tool will not act on it and we can either remove it or replace it with the sycl get_device() API.

## Build the `simpleCUBLAS_LU` Sample for CPU and GPU

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### On Linux*

1. Change to the sample directory.
2. For **Nvidia GPUs**, install the opensource dpcpp compiler & opensource oneMKL lib and set the environment variables before build.
   Here are the [steps](https://intel.github.io/llvm-docs/GetStartedGuide.html#:~:text=the%20same%20name.-,Linux,-%3A) to build opensource oneAPI DPC++ compiler & [steps to build oneMKL](https://oneapi-src.github.io/oneMKL/building_the_project.html#:~:text=install%20.%20%2D%2Dprefix%20%3Cpath_to_install_dir%3E-,Building%20for%20CUDA%C2%B6,-On%20Linux*)
   ```
   export PATH=path_to_opensource_DPC++_build_binaries:$PATH
   export LD_LIBRARY_PATH=path_to_opensource_DPC++_lib/:$LD_LIBRARY_PATH
   export ONEMKL_INSTALL_DIR=path_to_opensource_oneMKL_build_dir
   export LD_LIBRARY_PATH=$ONEMKL_INSTALL_DIR/lib:$LD_LIBRARY_PATH
   ```
4. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake .. or ( cmake -D INTEL_MAX_GPU=1 .. ) or ( cmake -D NVIDIA_GPU=1 .. ) or ( cmake -D FLOAT_TYPE=1 ..)
   $ make
   ```
>**Note:**
> - By default, no flags are enabled during the build which supports Intel® Gen9, Xeon CPU.
> - Enable INTEL_MAX_GPU flag during build which supports Intel® Data Center GPU Max 1550 or 1100 to get optimized performance.
> - Enable NVIDIA_GPU flag during build which supports NVIDIA GPUs.([oneAPI for NVIDIA GPUs plugin from Codeplay](https://developer.codeplay.com/products/oneapi/nvidia/)  is required to build for NVIDIA GPUs)
> - Enable FLOAT_TYPE flag as gen11 doesn't support double precision data type

By default, this command sequence will build the `sycl_migrated` versions of the program.

4. Run the code

   You can run the programs for CPU and GPU. The commands indicate the device target.

      Run `sycl_migrated` on GPU.
      ```
      make run_sm
      ```
      Run `sycl_migrated` on CPU.
      ```
      export ONEAPI_DEVICE_SELECTOR=opencl:cpu
      make run_sm
      unset ONEAPI_DEVICE_SELECTOR
      ```
      
### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/current/overview.html).

## License

Code samples are licensed under the MIT license. See
[License.txt](License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](third-party-programs.txt).
