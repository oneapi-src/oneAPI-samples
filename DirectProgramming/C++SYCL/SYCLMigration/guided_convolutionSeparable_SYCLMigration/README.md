# `convolutionSeparable` Sample

The convolution separable is a process in which a single convolution can be divided into two or more convolutions to produce the same output. The original CUDA* source code is migrated to SYCL for portability across GPUs from multiple vendors.

| Area              | Description
|:---                   |:---
| What you will learn              | Migrate convolutionSeparable from CUDA to SYCL and optimize
| Time to complete              | 15 minutes
| Category                      | Code Optimization

## Purpose

The sample shows the migration of convolutionSeperable from CUDA to SYCL using SYCLomatic tool and optimizing the migrated sycl code further to achieve good results.


>**Note**: We use Intel® open-sources SYCLomatic migration tool which assists developers in porting CUDA code automatically to SYCL code. To finish the process, developers complete the rest of the coding manually and then tune to the desired level of performance for the target architecture. Users can also use Intel® DPC++ Compatibility Tool which comes along with the Intel® oneAPI Base Toolkit.

This sample contains two versions in the following folders:

| Folder Name                   | Description
|:---                           |:---
| `01_dpct_output`              | Contains the output of SYCLomatic Tool which is a fully migrated version of CUDA code.
| `02_sycl_migrated_optimized`            | Contains the optimized sycl code

## Prerequisites

| Optimized for              | Description
|:---                   |:---
| OS                    | Ubuntu* 22.04
| Hardware              | Intel® Gen9 <br> Intel® Gen11 <br> Intel® Xeon CPU <br> Intel® Data Center GPU Max <br> NVIDIA Tesla P100 <br> NVIDIA A100 <br> NVIDIA H100
| Software                | SYCLomatic (Tag - 20240403) <br> Intel® oneAPI Base Toolkit version 2024.1 <br> oneAPI for NVIDIA GPUs" plugin from Codeplay (version 2024.1)

For more information on how to install Syclomatic Tool & DPC++ CUDA® plugin, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.v354cy) <br> How to run SYCL™ applications on NVIDIA® GPUs, refer to oneAPI for NVIDIA GPUs plugin from Codeplay [Install oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/)

## Key Implementation Details

This sample demonstrates the migration of the following CUDA features: 

- Shared memory
- Constant memory
- Cooperative groups
  
>  **Note**: Refer to [Workflow for a CUDA* to SYCL* Migration](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html#gs.s2njvh) for general information about the migration workflow.

### CUDA source code evaluation

A Separable Convolution is a process in which a single convolution can be divided into two or more convolutions to produce the same output. This sample implements a separable convolution filter of a 2D image with an arbitrary kernel. There are two functions in the code named convolutionRowsGPU and convolutionColumnsGPU in which the kernel functions (convolutionRowsKernel & convolutionColumnsKernel) are called where the loading of the input data and computations are performed. We validate the results with reference CPU separable convolution implementation by calculating the relative L2 norm.

This sample is migrated from the NVIDIA CUDA sample. See the sample [convolutionSeparable](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/convolutionSeparable) in the NVIDIA/cuda-samples GitHub.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Migrate the `convolutionSeparable` Sample

### Migrate the Code using SYCLomatic

For this sample, the SYCLomatic tool automatically migrates 100% of the CUDA runtime API's to SYCL. Follow these steps to generate the SYCL code using the compatibility tool:

1. Clone the required GitHub repository to your local environment.
   ```
   git clone https://github.com/NVIDIA/cuda-samples.git
   ```
2. Change to the convolutionSeparable sample directory.
   ```
   cd cuda-samples/Samples/2_Concepts_and_Techniques/convolutionSeparable/
   ```
3. Generate a compilation database with intercept-build
   ```
   intercept-build make
   ```
   The above step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.

4. Pass the JSON file as input to the Intel® SYCLomatic Compatibility Tool. The result is written to a folder named dpct_output. The --in-root specifies path to the root of the source tree to be migrated. The --gen-helper-function option will make a copy of dpct header files/functions used in the migrated code into the dpct_output folder as include folder.
   ```
   c2s -p compile_commands.json --in-root ../../.. --gen-helper-function
   ```
### Manual Workaround
CUDA code includes a custom API findCUDADevice in helper_cuda file to find the best CUDA Device available
```
 findCudaDevice (argc, (const char **) argv);   
```
Since its a custom API SYCLomatic tool will not act on it and we can either remove it or replace it with the sycl get_device() API.

### Optimizations

The migrated code can be optimized by using profiling tools which helps in identifying the hotspots (in this case convolutionRowsKernel() and convolutionColumnsKernel()).
 
If we observe the migrated SYCL code, especially in the above-mentioned function calls we see many ‘for’ loops that are being unrolled.
Although loop unrolling exposes opportunities for instruction scheduling optimization by the compiler and thus can improve performance, sometimes it may increase pressure on register allocation and cause register spilling. 

So, it is always a good idea to compare the performance with and without loop unrolling along with different times of unrolls to decide if a loop should be unrolled or how many times to unroll it.

In this case, by implementing the above technique, we can decrease the execution time by avoiding loop unrolling at the innermost “for-loop” of the computation part in convolutionRowsKernel function (line 120) and avoiding loop unrolling at the outer loop of the computation part in convolutionColumnsKernel function (line 242) of the file convolutionSeparable.dp.cpp.

Also, we can still decrease the execution time by avoiding the repetitive loading of c_Kernel[] array (as it is independent of `i` for-loop in convolutionSeparable.dp.cpp file). 

  ```
  for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
      float sum = 0;
  for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
     sum += c_Kernel[KERNEL_RADIUS - j] *s_Data[item_ct1.get_local_id(1)][item_ct1.get_local_id(2) + i * ROWS_BLOCKDIM_X + j];}
  ```

We can separate the array and load it into another new array and use it in place of c_Kernel

  ```
  float a[2*KERNEL_RADIUS + 1];
  for(int i=0; i<= 2*KERNEL_RADIUS; i++)
  a[i]=c_Kernel[i]; 
  ```
>**Note**: These optimization techniques also work with the larger input image sizes.

## Build the `convolutionSeparable` Sample for CPU and GPU

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
2. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake .. or ( cmake -D INTEL_MAX_GPU=1 .. ) or ( cmake -D NVIDIA_GPU=1 .. )
   $ make
   ```
>**Note:** 
> - By default, no flags are enabled during the build which supports Intel® UHD Graphics, Intel® Gen9, Gen11, Xeon CPU.
> - Enable INTEL_MAX_GPU flag during build which supports Intel® Data Center GPU Max 1550 or 1100 to get optimized performance.
> - Enable NVIDIA_GPU flag during build which supports NVIDIA GPUs.([oneAPI for NVIDIA GPUs plugin from Codeplay](https://developer.codeplay.com/products/oneapi/nvidia/)  is required to build for NVIDIA GPUs)
   
By default, this command sequence will build the `dpct_output` as well as `sycl_migrated_optimized` versions of the program.

3. Run the code

   You can run the programs for CPU and GPU. The commands indicate the device target.

      Run `dpct_output` on GPU.
      ```
      make run
      ```
      Run `dpct_output` on CPU.
      ```
      export ONEAPI_DEVICE_SELECTOR=opencl:cpu
      make run
      unset ONEAPI_DEVICE_SELECTOR
      ```
      Run `sycl_migrated_optimized` on GPU.
      ```
      make run_smo
      ```
      Run `sycl_migrated_optimized` on CPU.
      ```
      export ONEAPI_DEVICE_SELECTOR=opencl:cpu
      make run_smo
      unset ONEAPI_DEVICE_SELECTOR
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

