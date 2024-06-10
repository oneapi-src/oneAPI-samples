# `segmentationTreeThrust` Sample

This `segmentationTreeThrust` sample shows an approach to image segmentation trees construction. It is based on Boruvka's MST algorithm.
 
| Property                  | Description
|:---                       |:---
| What you will learn       | Migrating and optimizing segmentationTreeThrust from CUDA to SYCL
| Time to complete          | 15 minutes
| Category                      | Concepts and Functionality

## Purpose

The segmentationTreeThrust sample shows how to perform image segmentation using trees construction. It is a migrated version of a Nvidia CUDA sample. This README shows the process of migrating the sample

> **Note**: We use Intel® open-sources SYCLomatic tool which assists developers in porting CUDA code automatically to SYCL code. To finish the process, developers complete the rest of the coding manually and then tune to the desired level of performance for the target architecture. User's can also use SYCLomatic Tool which comes along with the Intel® oneAPI Base Toolkit.

This sample contains two versions in the following folders:

| Folder Name                   | Description
|:---                           |:---
| `01_dpct_output`              | Contains the output of the SYCLomatic tool used to migrate SYCL-compliant code from CUDA code. The tool completely migrates code but needs manual changes to get functional correctness on the given list of hardware.
| `02_sycl_migrated`            | Contains migrated SYCL code from CUDA code with manual changes.

## Prerequisites

| Optimized for              | Description
|:---                        |:---
| OS                         | Ubuntu* 22.04
| Hardware              | Intel® Gen9 <br> Intel® Gen11 <br> Intel® Xeon CPU <br> Intel® Data Center GPU Max <br> NVIDIA Tesla P100 <br> NVIDIA A100 <br> NVIDIA H100
| Software                | SYCLomatic (Tag - 20231004) <br> Intel® oneAPI Base Toolkit version 2024.0.0 <br> oneAPI for NVIDIA GPUs" plugin from Codeplay (version 2024.0)

For more information on how to install Syclomatic Tool & DPC++ CUDA® plugin, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.v354cy) <br> How to run SYCL™ applications on NVIDIA® GPUs, refer to oneAPI for NVIDIA GPUs plugin from Codeplay [Install oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/)

## Key Implementation Details

This sample demonstrates the migration of the following prominent CUDA features: 
- Cooperative Groups
- Data-Parallel Algorithms
- Performance Strategies

>  **Note**: Refer to [Workflow for a CUDA* to SYCL* Migration](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html#gs.s2njvh) for general information about the migration workflow.

### CUDA source code evaluation

This sample is migrated from NVIDIA CUDA sample. See the [segmentationTreeThrust](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/segmentationTreeThrust) sample in the NVIDIA/cuda-samples GitHub.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Migrate the `threadFence segmentationTreeThrust` Sample

### Migrate the Code using SYCLomatic

For this sample, the SYCLomatic Tool automatically migrates 100% of the CUDA code to SYCL. Follow these steps to generate the SYCL code using the compatibility tool:

1. git clone https://github.com/NVIDIA/cuda-samples.git
2. cd cuda-samples/Samples/2_Concepts_and_Techniques/segmentationTreeThrust/
3. Generate a compilation database with intercept-build
   ```
   intercept-build make
   ```
4. The above step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.
5. Pass the JSON file as input to the Intel® SYCLomatic Compatibility Tool. The result is written to a folder named dpct_output. The --in-root specifies path to the root of the source tree to be migrated.
   ```
   c2s -p compile_commands.json --in-root ../../.. --gen-helper-function
   ```
### Manual Workaround
CUDA code includes a custom API findCUDADevice in helper_cuda file to find the best CUDA Device available
```
 findCudaDevice (argc, (const char **) argv);   
```
Since its a custom API SYCLomatic tool will not act on it and we can either remove it or replace it with the sycl get_device() API.

## Build the `segmentationTreeThrust` Sample for CPU and GPU

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

   By default, this command sequence will build the `02_sycl_migrated` versions of the program.

3. Run the program.
   
   Run `02_sycl_migrated` on GPU.
   ```
   make run
   ```  
   Run `02_sycl_migrated` on CPU.
   ```
   export ONEAPI_DEVICE_SELECTOR=opencl:cpu
   make run
   unset ONEAPI_DEVICE_SELECTOR 
   ```
5. Running the executable with command line arguments
   ```
   --shmoo:         Test performance for 1 to 32M elements with each of the 7 different kernels <br>
   --n=<N>:         Specify the number of elements to reduce (default 1048576) <br>
   --threads=<N>:   Specify the number of threads per block (default 128) <br>
   --maxblocks=<N>: Specify the maximum number of thread blocks to launch (kernel 6 only, default 64) <br>
   --cpufinal:      Read back the per-block results and do final sum of block sums on CPU (default false) <br>
   --cputhresh=<N>: The threshold of number of blocks sums below which to perform a CPU final reduction (default 1) <br>
   --multipass:     Use a multipass reduction instead of a single-pass reduction
   ```
    For example, to change the number of elements to reduce using the comment line argument.
    ```
     ./bin/02_sycl_migrated --n=2097152 
    ```
  
#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/2024-0/overview.html) for more information on using the utility.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
