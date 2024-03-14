# `Simple Atomic Intrinsics` Sample

The `Simple Atomic Intrinsics` sample demonstrates the use of various SYCL arithmetic Atomic Intrinsic functions. The original CUDA* source code is migrated to SYCL for portability across GPUs from multiple vendors.

| Area                   | Description
|:---                    |:---
| What you will learn    | How to begin migrating CUDA to SYCL
| Time to complete       | 15 minutes
| Category               | Concepts and Functionality

>**Note**: This sample is migrated from the NVIDIA CUDA sample. See the [SimpleAtomicIntrinsics](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/simpleAtomicIntrinsics) sample in the NVIDIA/cuda-samples GitHub.

## Purpose

The `Simple Atomic Intrinsics` sample shows the execution of multiple atomic intrinsic functions on the device.

> **Note**: The sample used the open-source SYCLomatic tool that assists developers in porting CUDA code to SYCL code. To finish the process, you must complete the rest of the coding manually and then tune to the desired level of performance for the target architecture. You can also use the Intel® DPC++ Compatibility Tool available to augment the Base Toolkit.

This sample contains two versions of the code in the following folders:

| Folder Name          | Description
|:---                  |:---
|`01_dpct_output`      | Contains the output of SYCLomatic Tool used to migrate SYCL-compliant code from CUDA code. 
|`02_sycl_migrated`    | Contains migrated and cleaned-up SYCL code from CUDA code.

## Prerequisites

| Optimized for         | Description
|:---                   |:---
| OS                    | Ubuntu* 22.04
| Hardware              | Intel® Gen9 <br>Intel® Gen11 <br>Intel® Xeon CPU <br>Intel® Data Center GPU Max <br> Nvidia Tesla P100 <br> Nvidia A100 <br> Nvidia H100 
| Software              | SYCLomatic (Tag - 20240116) <br> Intel® oneAPI Base Toolkit (Base Kit) version 2024.0 <br> oneAPI for NVIDIA GPU plugin(version 2024.0) from Codeplay (to run SYCL™ applications on NVIDIA® GPUs)

For more information on how to install Syclomatic Tool & DPC++ CUDA® plugin, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.v354cy) .<br>
How to run SYCL™ applications on NVIDIA® GPUs, refer to 
[oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/) plugin from Codeplay.

## Key Implementation Details

This sample demonstrates the migration of the following prominent CUDA features:

- Atomic Intrinsics

The kernel `testKernel` demonstrates SYCL arithmetic atomic functions in device code such as `atomic_fetch_add`, `atomic_fetch_sub`, `atomic_exchange`, `atomic_fetch_max`, `atomic_fetch_min`, `atomic_compare_exchange_strong`, `atomic_fetch_and`, `atomic_fetch_or`, and `atomic_fetch_xor` migrated from CUDA atomic instructions.

>**Note**: The DPC++ compiler is currently in the process of incorporating native support for atomic increment/decrement operations, along with ongoing performance enhancements.
>**Note**: Refer to [Workflow for a CUDA* to SYCL* Migration](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html) for general information about the migration workflow.

## CUDA source code evaluation
The `Simple Atomic Intrinsics` CUDA sample demonstrates the use of global memory atomic instructions.

> **Note**: For more information on how to use the Syclomatic Tool, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.vmhplg).

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Migrate the Code Using SYCLomatic

For this sample, the SYCLomatic tool automatically migrates 100% of the CUDA code to SYCL. Follow these steps to generate the SYCL code using the compatibility tool.

1. Clone the required GitHub repository to your local environment.
   ```
   git clone https://github.com/NVIDIA/cuda-samples.git
   ```
2. Change to the SimpleAtomicIntrinsics sample directory.
   ```
   cd cuda-samples/Samples/0_Introduction/simpleAtomicIntrinsics/
   ```
3. Generate a compilation database with intercept-build.
   ```
   intercept-build make
   ```
   This step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.

4. Pass the JSON file as input to the Intel® SYCLomatic Compatibility Tool. The result is written to a folder named dpct_output. The `--in-root` specifies the path to the root of the source tree to be migrated. The `--gen-helper-function`  option will make a copy of the dpct header files/functions used in migrated code into the dpct_output folder as include folder. 
   ```
   c2s -p compile_commands.json --in-root ../../.. --gen-helper-function 
   ```
   
## Build and Run the `Simple Atomic Intrinsics` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script at the root of your oneAPI installation.
>
> Linux*:
> - For system-wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files (x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*.

### On Linux*

1. Change to the sample directory.
2. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake .. or ( cmake -D INTEL_MAX_GPU=1 .. ) or ( cmake -D NVIDIA_GPU=1 .. )
   $ make
   ```
   > **Note**:
   > - By default, no flags are enabled during the build which supports Intel® UHD Graphics, Intel® Gen9, Gen11, and Xeon CPU.
   > - Enable **INTEL_MAX_GPU** flag during build which supports Intel® Data Center GPU Max 1550 or 1100 to get optimized performance.
   > - Enable **NVIDIA_GPU** flag during the build which supports NVIDIA GPUs([oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/) plugin from 
   Codeplay is required to build for NVIDIA GPUs).
   >   
   >

   
   By default, this command sequence will build the  `01_dpct_output` and `02_sycl_migrated` version of the program.
3. Run `01_dpct_output` on CPU and GPU.
   
   ```
   make run_cpu (runs on CPU)
   make run_gpu (runs on Level-Zero Backend)
   make run_gpu_opencl (runs on OpenCL Backend)
   make run_gpu_cuda (runs on cuda Backend)
   ```
4. Run `02_sycl_migrated` on CPU and GPU.
   
   ```
   make run_sm_cpu (runs on CPU)
   make run_sm_gpu (runs on Level-Zero Backend)
   make run_sm_gpu_opencl (runs on OpenCL Backend)
   make run_sm_gpu_cuda (runs on cuda Backend)
   ```

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
