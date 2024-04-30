# `BandwidthTest` Sample

The `bandwidthTest` sample measure the memcopy bandwidth of the GPU and memcpy bandwidth across PCI-e. This test application is capable of measuring device-to-device copy bandwidth, host-to-device copy bandwidth for pageable and page-locked memory, and device-to-host copy bandwidth for pageable and page-locked memory.
The original CUDA* source code is migrated to SYCL for portability across GPUs from multiple vendors.

| Area                   | Description
|:---                    |:---
| What you will learn    | How to migrate and map CUDA Runtime API to SYCL API 
| Time to complete       | 15 minutes
| Category               | Concepts and Functionality

>**Note**: This sample is migrated from NVIDIA CUDA sample. See the [bandwidthTest](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/1_Utilities/bandwidthTest) sample in the NVIDIA/cuda-samples GitHub.

## Purpose

This is a simple test program to measure the memcopy bandwidth of the GPU. It can measure device-to-device copy bandwidth, host-to-device copy bandwidth
for pageable and pinned memory and device to host copy bandwidth for pageable and pinned memory. 

> **Note**: The sample used the open-source [SYCLomatic tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html) that assists developers in porting CUDA code to SYCL code. To finish the process, you must complete the rest of the coding manually and then tune to the desired level of performance for the target architecture. You can also use the [Intel® DPC++ Compatibility Tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compatibility-tool.html#gs.5g2aqn) available to augment Base Toolkit.

This sample contains two versions of the code in the following folders:

| Folder Name          | Description
|:---                  |:---
|`01_dpct_output`      | Contains the output of SYCLomatic Tool used to migrate SYCL-compliant code from CUDA code. This SYCL code has full functionality as generated.
|`02_sycl_migrated`    | Contains manually migrated SYCL code from CUDA code.

## Prerequisites

| Optimized for         | Description
|:---                   |:---
| OS                    | Ubuntu* 22.04
| Hardware              | Intel® Gen9 <br>Intel® Gen11 <br>Intel® Xeon CPU <br>Intel® Data Center GPU Max <br> Nvidia Tesla P100 <br> Nvidia A100 <br> Nvidia H100 
| Software              | SYCLomatic (Tag - 20240403) <br> Intel® oneAPI Base Toolkit (Base Kit) version 2024.1 <br> oneAPI for NVIDIA GPU plugin from Codeplay (to run SYCL™ applications on NVIDIA® GPUs)

For information on how to use SYCLomatic, refer to the materials at *[Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html)*.<br> How to run SYCL™ applications on NVIDIA® GPUs, refer to 
[oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/) plugin from Codeplay.


## Key Implementation Details
This sample demonstrates the migration of the following prominent CUDA features:
 - CUDA API (cudaHostAlloc, cudaMemcpy, cudaMalloc, cudaMemcpyAsync, cudaFree, cudaGetErrorString, cudaMallocHost, cudaSetDevice, cudaGetDeviceProperties, cudaDeviceSynchronize, cudaEventRecord, cudaFreeHost, cudaEventDestroy, cudaEventElapsedTime, cudaGetDeviceCount, cudaEventCreate)

BandwidthTest sample demonstrates how to measure the memcopy bandwidth of the GPU. It can measure device-to-device copy bandwidth, host-to-device copy bandwidth
for pageable and pinned memory and device to host copy bandwidth for pageable and pinned memory.

>**Note**: Refer to [Workflow for a CUDA* to SYCL* Migration](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html) for general information about the migration workflow.
## CUDA source code evaluation

The BandwidthTest sample demonstrates how to measure the memcopy bandwidth of the GPU. 

- Measure device-to-device copy bandwidth.
- Measure host-to-device copy bandwidth for pageable and pinned memory.
- Measure device-to-host copy bandwidth for pageable and pinned memory.


> **Note**: For more information on how to use Syclomatic Tool, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.vmhplg).

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.


## Migrate the Code Using SYCLomatic

For this sample, the SYCLomatic tool automatically migrates 100% of the CUDA code to SYCL. Follow these steps to generate the SYCL code using the compatibility tool.

1. Clone the required GitHub repository to your local environment.
   ```
   git clone https://github.com/NVIDIA/cuda-samples.git
   ```
2. Change to the bandwidthTest sample directory.
   ```
   cd cuda-samples/Samples/1_Utilities/bandwidthTest/
   ```
3. Generate a compilation database with intercept-build.
   ```
   intercept-build make
   ```
   This step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.

4. Pass the JSON file as input to the Intel® SYCLomatic Compatibility Tool. The result is written to a folder named dpct_output. The --in-root specifies the path to the root of the source tree to be migrated.
   ```
   c2s -p compile_commands.json --in-root ../../.. --gen-helper-function 
   ```

## Build and Run the `BandwidthTest` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
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
   > - By default, no flags are enabled during build which supports Intel® UHD Graphics, Intel® Gen9, Gen11, Xeon CPU.
   > - Enable **INTEL_MAX_GPU** flag during build which supports Intel® Data Center GPU Max 1550 or 1100 to get optimized performance.
   > - Enable **NVIDIA_GPU** flag during build which supports NVIDIA GPUs.([oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/) plugin   from Codeplay is required to build for NVIDIA GPUs )

 
   By default, this command sequence will build the `01_dpct_output` and `02_sycl_migrated` versions of the program.

3. Run `01_dpct_output` on GPU.
   ```
   make run
   ```
   Run `01_dpct_output` on CPU.
   
   ```
   export ONEAPI_DEVICE_SELECTOR=opencl:cpu
   make run
   unset ONEAPI_DEVICE_SELECTOR
   ``` 
4. Run `02_sycl_migrated` on GPU.
   ```
   make run_sm
   ```
   Run `02_sycl_migrated` on CPU.
   ```
   export ONEAPI_DEVICE_SELECTOR=opencl:cpu
   make run_sm
   unset ONEAPI_DEVICE_SELECTOR
   ```
5. Running the executable with command line arguments
    ```
   ./bin/02_sycl_migrated --help
    [SYCL Bandwidth Test] - Starting...
     Usage:  bandwidthTest [OPTION]...
     Test the bandwidth for device to host, host to device, and device to device transfers
     
     Example:  measure the bandwidth of device to host pinned memory copies in the range 1024 Bytes to 102400 Bytes in 1024 Byte increments
     ./bandwidthTest --memory=pinned --mode=range --start=1024 --end=102400 --increment=1024 --dtoh
     
     Options:
     --help  Display this help menu
     --csv   Print results as a CSV
     --device=[deviceno]     Specify the device device to be used
       all - compute cumulative bandwidth on all the devices
       0,1,2,...,n - Specify any particular device to be used
     --memory=[MEMMODE]      Specify which memory mode to use
       pageable - pageable memory
       pinned   - non-pageable system memory
     --mode=[MODE]   Specify the mode to use
       quick - performs a quick measurement
       range - measures a user-specified range of values
       shmoo - performs an intense shmoo of a large range of values
     --htod  Measure host to device transfers
     --dtoh  Measure device to host transfers
     --dtod  Measure device to device transfers
     --wc    Allocate pinned memory as write-combined
     --cputiming     Force CPU-based timing always
     Range mode options
     --start=[SIZE]  Starting transfer size in bytes
     --end=[SIZE]    Ending transfer size in bytes
     --increment=[SIZE]      Increment size in bytes
   ```
    For example, to get only the pageable memory bandwidth using the comment line argument. 

   ```
    ./bin/02_sycl_migrated --memory=pageable
    ```
#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/2024-1/overview.html) for more information on using the utility.


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
