# `American Options` Sample

American Options Pricing is a model that is based on the Monte Carlo method and widely used in financial services industry.
The original [CUDA* source code](https://github.com/NVIDIA-developer-blog/code-samples/tree/master/posts/american-options) is migrated to SYCL for portability across GPUs from multiple vendors.

| Area                       | Description
|:---                        |:---
| What you will learn        | Migrate American Options from CUDA to SYCL and optimize
| Time to complete           | 10 minutes or less
| Category                   | Code Optimization

## Purpose

The sample shows the migration of American Options from CUDA to SYCL
using Intel® DPC++ Compatibility Tool tool and optimizing the migrated SYCL code
further to achieve better results. Results are used for [the article](https://www.intel.com/content/www/us/en/developer/articles/technical/onemkl-random-number-generator-device-routines.html).

This sample contains two versions in the following folders:

| Folder Name                   | Description
|:---                           |:---
| `01_dpct_output`              | Contains the output of Intel® DPC++ Compatibility Tool Tool which is a fully migrated version of CUDA code, but it has compilation issues.
| `02_sycl_migrated_optimized`  | Contains the optimized SYCL code that can compile and run.

## Prerequisites

| Optimized for         | Description
|:---                   |:---
| OS                    | Ubuntu* 22.04
| Hardware              | Intel® Gen11 <br> Intel® Xeon CPU <br> Intel® Data Center GPU Max
| Software              | Intel® oneAPI Base Toolkit version 2024.0.0

## Key Implementation Details

This sample demonstrates the migration of the following CUDA features: 

- Shared memory

>  **Note**: Refer to [Workflow for a CUDA* to SYCL* Migration](
https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html) for general information about the migration workflow.

### CUDA source code evaluation

This sample is migrated from the NVIDIA CUDA sample.
See the [american options sample](
https://github.com/NVIDIA-developer-blog/code-samples/tree/master/posts/american-options)
in the `NVIDIA-developer-blog/code-samples` GitHub.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the
oneAPI toolkits using environment variables. Set up your CLI environment by
sourcing the `setvars` script every time you open a new terminal window. This
practice ensures that your compiler, libraries, and tools are ready for development.

## Migrate the `American Options` Code

### Changes that were done for the original CUDA code

Following changes are needed for better alignment with desire outcome.

1. Add timers that are aligned with measurement in C++ code.
2. Replace the legacy `__any` function with a newer one: `__any_sync`.
3. Seed is equal 0.
4. Simulation parameters are aligned with desired parameters.
5. Add warm up call of the `do_run` function to not measure the first call
   that can be slower than other iterations.

### Migrate the Code using Intel® DPC++ Compatibility Tool

Follow these steps to generate the SYCL code using Intel® DPC++ Compatibility Tool:

1. Clone the required GitHub repository to your local environment.
   ```
   git clone https://github.com/NVIDIA-developer-blog/code-samples.git
   ```
2. Change to the code directory.
   ```
   cd code-samples/posts/american-options/
   ```
3. Call DPC++ Compatibility Tool. The result is written to a folder named `dpct_output`.
   `--extra_arg` specify additional argument to append to the migration command line
   since we are interested in a case with `WITH_FUSED_BETA == 1`.
   `--cuda-include-path` is the directory path of the CUDA header files if needed.
   ```
   dpct longstaff_schwartz_svd_2.cu --cuda-include-path=<CUDA_PATH>/include --extra-arg="-DWITH_FUSED_BETA=1"
   ```

### Changes that were done for the migrated code

The migrated code can be found in the `01_dpct_output` folder. However, it doesn't compile.
Some changes need to be done to make the compilation happen and to make it works
correctly.

1. The original CUDA code uses CUB API. There are analogues in SYCL that
   Intel® DPC++ Compatibility Tool successfully substitutes. However, there are
   some artefacts like `typename <...>::TempStorage` that are not really used in
   the code, but compiler triggers. So, `TempStorage` and variables that depend on it
   need to be removed.
2. Add timers for performance measurements of RNG steps and the whole benchmark..
3. Add `wait()` call at the end of `do_run` to have a synchronization point.
4. Add `wait()` after `generate_paths_kernel` to measure RNG performance data precisely.
5. Change `get_pointer` to `get_multi_ptr` according to SYCL 2020 changes.
6. Add `NUM_PATHS`, `NUM_TIMESTEPS` macros to scale tasks. This is relevant
   for the article measurements.
7. Replace `generate_gaussian` with the oneMKL `generate` call.

### Optimizations using RNG Device API

To measure performance with RNG Device API following changes need to be introduced:
1. Macro `USE_DEVICE_API` is to distinguish between implementation for Device API and Host API.
2. Remove memory allocation that used to store random numbers.
3. Introduce Device API calls in `generate_paths_kernel` because random numbers can
   be generated within the SYCL kernel and can be used immediately to generate paths.
4. Do not call RNG Host API as a separate call.

## Build the `American Options` Code for CPU and GPU

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
   $ cmake .. or ( cmake -D USE_DEVICE_API=1 .. )
   $ make
   ```
>**Note:** 
> - By default, no flags are enabled during the build which supports
    Intel® UHD Graphics, Intel® Gen11, Xeon CPU.
> - Enable the `USE_DEVICE_API` flag during build to run using Intel oneMKL.
   
By default, this command sequence will build the `dpct_output` as well as `sycl_migrated_optimized` versions of the program.

3. Run the code

   You can run the programs for CPU and GPU. The commands indicate the device target.

      Run `sycl_migrated_optimized` on GPU.
      ```
      make run_host_api # or make run_device_api for RNG Device API usage
      ```
      Run `sycl_migrated_optimized` on CPU.
      ```
      export ONEAPI_DEVICE_SELECTOR=opencl:cpu
      make run_host_api # or make run_device_api for RNG Device API usage
      unset ONEAPI_DEVICE_SELECTOR
      ```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/current/overview.html).

## License

Code samples are licensed under the MIT license. See
[License.txt](License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](third-party-programs.txt).
