# `SHFL_Scan` Sample
 
The `SHFL_Scan`, CUDA parallel prefix sum with shuffle intrinsics sample demonstrates the use of shuffle intrinsic __shfl_up_sync to perform a scan operation across a thread block. The sample also demonstrates the migration of these CUDA shuffle intrinsic APIs to the SYCL group algorithm. The original CUDA* source code is migrated to SYCL for portability across GPUs from multiple vendors.

| Area                      | Description
|:---                       |:---
| What you will learn       | Migrate the SHFL_Scan sample from CUDA to SYCL.
| Time to complete          | 15 minutes
| Category                  | Concepts and Functionality

> **Note**: This sample is based on the [SHFL_Scan](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/shfl_scan) sample in the NVIDIA/cuda-samples GitHub repository.

## Purpose

The parallel implementation demonstrates the use of shuffle intrinsic __shfl_up_sync to perform a scan operation across a thread block.
It covers explanations of key SYCL concepts, such as

- Group algorithm
- Shared Memory

>  **Note**: The sample used the open-source [SYCLomatic tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html) that assists developers in porting CUDA code to SYCL code. To finish the process, you must complete the rest of the coding manually and then tune to the desired level of performance for the target architecture. You can also use the [Intel® DPC++ Compatibility Tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compatibility-tool.html#gs.5g2aqn) available to augment Base Toolkit.

This sample contains two versions in the following folders:

| Folder Name                             | Description
|:---                                     |:---
| `01_dpct_output`                        | Contains the output of the SYCLomatic tool used to migrate SYCL-compliant code from CUDA code. The tool completely migrates code but needs manual changes to get functional correctness on the given list of hardware.
| `02_sycl_migrated`                      | Contains migrated SYCL code from CUDA code with manual changes.

## Prerequisites

| Optimized for              | Description
|:---                        |:---
| OS                         | Ubuntu* 22.04
| Hardware                   | Intel® Gen9 <br> Intel® Gen11 <br> Intel® Xeon CPU <br> Intel® Data Center GPU Max <br> Nvidia Tesla P100 <br> Nvidia A100 <br> Nvidia H100 
| Software                   | SYCLomatic (Tag - 20240403) <br> Intel® oneAPI Base Toolkit (Base Kit) version 2024.1 <br> oneAPI for NVIDIA GPUs plugin (version 2024.1) from Codeplay

For more information on how to install Syclomatic Tool, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.1k3c5h) <br>
Refer to [oneAPI for NVIDIA GPUs plugin](https://developer.codeplay.com/products/oneapi/nvidia/) from Codeplay to execute a sample on NVIDIA GPU.

## Key Implementation Details

This sample demonstrates the migration of the following prominent CUDA features:

- Warp-level Primitives
- Shared Memory

The computation of `shuffle_simple_test` host method is included in two kernels, `shfl_scan_test` where __shfl_up is used to perform a scan operation across a block. It performs a scan inside a warp then to continue the scan operation across the block, each warp's sum is placed into shared memory. A single warp then performs a shuffle scan on that shared memory these results are then uniformly added to each warp's threads. The final sum of each block is then placed in global memory and the prefix sum is computed by the `uniform_add` kernel call.

In the `shuffle_integral_image_test` method each thread is set to handle 16 values. In horizontal scan `get_prefix_sum` kernel the prefix sum for each thread's 16 value is computed, and the final sums are shared with other threads through the __shfl_up() instruction and a shuffle scan operation is performed to distribute the sums to the correct threads. Then shuffle `__shfl_xor` command is used to reformat the data inside the registers so that each thread holds consecutive data to be written so larger contiguous segments can be assembled for writing. 

In vertical scan, prefix sums are computed column-wise. The approach here is to have each block compute a local set of sums. So first, the data covered by the block is loaded into shared memory, then instead of performing a sum in shared memory using __syncthreads, between stages, the data is reformatted so that the necessary sums occur inside warps and the shuffle scan operation is used. The final set of sums from the block is then propagated, with the block computing "down" the image and adding the running sum to the local block sums.

>  **Note**: Refer to [Workflow for a CUDA* to SYCL* Migration](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html#gs.s2njvh) for general information about the migration workflow.

### CUDA Source Code Evaluation

The SHFL_Scan CUDA sample includes two implementations of scan operation using the shuffle intrinsic operation.
   1. `shuffle_simple_test` Method
   2. `shuffle_integral_image_test` Method

In the shuffle_simple_test method, a prefix sum is computed using the shuffle intrinsic for 65536 number of elements. This implementation is verified with the result from serial implementation i.e., `CPUverify` method.
In the shuffle_integral_image_test method, computation of an integral image using the shuffle intrinsic is provided for 1920x1080 number of elements, where shuffle scan operation and shuffle xor operations are used. This method includes two approaches:
   i. A horizontal (scanline) / Fast scan
   ii. A vertical (column) scan  

The result from the horizontal scan is verified by comparing the result from `verifyDataRowSums` serial implementation, and the vertical scan result is verified by checking the final value in the corner which must be as same as the size of the buffer.

For information on how to use SYCLomatic, refer to the materials at *[Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html)*.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Migrate the `SHFL_Scan` Sample

### Migrate the Code using SYCLomatic

For this sample, the SYCLomatic tool automatically migrates 100% of the CUDA runtime APIs to SYCL. Follow these steps to generate the SYCL code using the compatibility tool.

1. Clone the required GitHub repository to your local environment.
   ```
   git clone https://github.com/NVIDIA/cuda-samples.git
   ```
2. Change to the shfl_scan sample directory.
   ```
   cd cuda-samples/Samples/2_Concepts_and_Techniques/shfl_scan/
   ```
3. Generate a compilation database with intercept-build
   ```
   intercept-build make
   ```
   The above step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.

4. Pass the JSON file as input to the SYCLomatic tool. The result is written to a folder named dpct_output. The `--in-root` specifies the path to the root of the source tree to be migrated. The `--gen-helper-function` option will make a copy of the dpct header files/functions used in migrated code into the dpct_output folder as `include` folder. The `--use-experimental-features` option specifies an experimental helper function used to logically group work-items.
   ```
   c2s -p compile_commands.json --in-root ../../.. --gen-helper-function --use-experimental-features=logical-group
   ```

### Manual Workarounds 

1. CUDA code includes a custom API `findCUDADevice` in the helper_cuda file to find the best CUDA Device available.
```
    findCudaDevice (argc, (const char **) argv);
```
Since it's a custom API SYCLomatic tool will not act on it and we can either remove it or replace it with the `dpct get_device()` API to get device details.

2. CUDA code includes an `if condition` to check for required CUDA computability as CUDA __shfl intrinsic needs SM 3.0 or higher which is CUDA device specific and different from the SYCL device version. We can either rewrite the code or remove it.
```
    if (deviceProp.major < 3) {
    printf("> __shfl() intrinsic requires device SM 3.0+\n");
    printf("> Waiving test.\n");
    exit(EXIT_WAIVED);
  }
```

## Build and Run the `SHFL_Scan` Sample

>  **Note**: If you have not already done so, set up your CLI
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
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*

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
   > - By default, no flags are enabled during the build which supports Intel® UHD Graphics, Intel® Gen9, Gen11, Xeon CPU. <br>
   > - Enable **INTEL_MAX_GPU** flag during build which supports Intel® Data Center GPU Max 1550 or 1100 to get optimized performance. <br>
   > - Enable **NVIDIA_GPU** flag during the build which supports NVIDIA GPUs.([oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/) plugin   from Codeplay is required to build for NVIDIA GPUs ) <br>

   By default, this command sequence will build the `02_sycl_migrated` version of the program.
   
3. Run the program.
   
   Run `02_sycl_migrated` on GPU.
   ```
   $ make run
   ```   
   Run `02_sycl_migrated` for CPU.
    ```
    $ export ONEAPI_DEVICE_SELECTOR=opencl:cpu
    $ make run
    $ unset ONEAPI_DEVICE_SELECTOR
    ```
   
#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
$ make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/2024-1/overview.html) for more information on using the utility.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
