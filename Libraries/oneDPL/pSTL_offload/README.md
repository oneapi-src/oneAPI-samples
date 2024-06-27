# `pSTL offload` Sample
 
The `pSTL_offload` sample demonstrates the offloading of C++ standard parallel algorithms to a SYCL device. 

| Area                      | Description
|:---                       |:---
| What you will learn       | Offloading of C++ standard algorithms to GPU devices. 
| Time to complete          | 15 minutes
| Category                  | Concepts and Functionality

> **Note**: This sample is based on the [cppParallelSTL](https://github.com/vladiant/CppParallelSTL) GitHub repository.

## Purpose

Offloading the C++ standard parallel STL code (par-unseq policy) to GPU and CPU  without any code changes when using the `-fsycl-pstl-offload` compiler option with Intel® DPC+/C+ compiler. It is an experimental feature of oneDPL.

This folder contains two sample examples in the following folders:

| Folder Name                           | Description
|:---                                   |:---
| `FileWordCount`                       | Counting Words in Files Example
| `WordCount`                           | Counting Words generated Example

> **Note**: For more information refer to [Get Started with Parallel STL](https://www.intel.com/content/www/us/en/developer/articles/guide/get-started-with-parallel-stl.html).


## Prerequisites

| Optimized for                                      | Description
|:---                                                |:---
| OS                                                 | Ubuntu* 22.04
| Hardware                                           | Intel® Data Center GPU Max <br> Intel® Xeon CPU <br>
| Software                                           | Intel oneAPI Base Toolkit version 2024.2 <br> Intel® Threading Building Blocks (Intel® TBB)

## Key Implementation Details

The example includes two samples `FileWordCount` and `WordCount` which count the number of words in files and the number of words generated respectively using the standard C++17 Parallel Algorithm [transfor_reduce](https://en.cppreference.com/w/cpp/algorithm/transform_reduce). This computation can be offloaded to the GPU device with the help of `-fsycl-pstl-offload` compiler option and standard <algorithm> header inclusion is explicitly required for PSTL Offload to work.
FileWordCount sample also demonstrates the use of transform, copy, copy_if, and for_each standard C++17 Parallel Algorithms.
The `-fsycl-pstl-offload` option enables the offloading of C++ standard parallel algorithms that were only called with `std::execution::par_unseq` policy to a SYCL device. The offloaded algorithms are implemented via the oneAPI Data Parallel C++ Library (oneDPL). This option is an experimental feature. If the argument is not specified, the compiler offloads to the default SYCL device.
The performance of memory allocations may be improved by using the `SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR` environment variable.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build and Run the `pSTL offload` Samples

>  **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script at the root of your oneAPI installation.
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
   $ ( cmake -D GPU=1 .. ) or ( cmake -D CPU=1 .. )
   $ make
   ```

   **Note**: Enable **GPU** flag during the build which supports Intel® Data Center GPU Max 1550 or 1100 to execution on GPUs. <br>
    Enable **CPU** flag during the build to execution on GPU. <br>

   This command sequence will build the `WordCount` and `FileWordCount` samples.
   
3. Run the program.
   
   Run `pSTL_offload-WordCount` on GPU.
   ```
   $ export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
   $ make run_wc
   $ unset ONEAPI_DEVICE_SELECTOR
   ```
   Run `pSTL_offload-WordCount` on CPU.
   ```
   $ export ONEAPI_DEVICE_SELECTOR=*:cpu
   $ make run_wc
   $ unset ONEAPI_DEVICE_SELECTOR
   ```

   Run `pSTL_offload-FileWordCount` on GPU.
   ```
   $ export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
   $ make run_fwc0               //for SEQ Policy
   $ make run_fwc1               //for PAR Policy
   $ unset ONEAPI_DEVICE_SELECTOR
   ```
    
   Run `pSTL_offload-FileWordCount` on CPU.
    ```
    $ export ONEAPI_DEVICE_SELECTOR=*:cpu
    $ make run_fwc0              //for SEQ Policy
    $ make run_fwc1              //for PAR Policy
    $ unset ONEAPI_DEVICE_SELECTOR
    ```
    
#### Troubleshooting

If an error occurs, you can get more details by running `make` with the `VERBOSE=1` argument:
```
$ make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/2024-1/overview.html) for more information on using the utility.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
