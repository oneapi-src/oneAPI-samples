# `Odd-Even MergeSort` Sample

The `Odd-Even MergeSort` sample demonstrates how to use the **odd-even mergesort** algorithm (also known as "Batcher's odd–even mergesort"), which belongs to the class of sorting networks. Generally, this algorithm is not efficient for large sequences compared to algorithms with better asymptotic algorithmic complexity (merge sort or radix sort); however, this sorting method might be the preferred algorithm for sorting batches of short-sized to mid-sized (key, value) array pairs.

| Area                | Description
|:---                 |:---
| What you will learn | How to begin migrating CUDA to SYCL
| Time to complete    | 15 minutes
| Category            | Concepts and Functionality

> **Note**: This sample is migrated from the NVIDIA CUDA sample. See the [sortingNetworks](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/sortingNetworks) sample in the NVIDIA/cuda-samples GitHub.

## Purpose

The **odd-even mergesort** algorithm was developed by Kenneth Edward Batcher.

The algorithm is based on a merge algorithm that merges two sorted halves of a sequence into a sorted sequence. In contrast to a standard merge sort, this algorithm is not data-dependent. All comparisons are performed regardless of the actual data, so you can implement this algorithm as a sorting network.

> **Note**: We use Intel® open-sources SYCLomatic tool, which assists developers in porting CUDA code automatically to SYCL code. To finish the process, developers complete the rest of the coding manually and then tune to the desired level of performance for the target architecture. You can use SYCLomatic tool that comes along with the Intel® oneAPI Base Toolkit.

This sample contains two versions of the program.

| Folder Name             | Description
|:---                     |:---
| `01_dpct_output`        | Contains the output of SYCLomatic Tool used to migrate SYCL-compliant code from CUDA code. This SYCL code has some code that is not migrated and has to be manually fixed to get full functionality (The code does not functionally work as supplied).
| `02_sycl_migrated`      | Contains manually migrated SYCL code from CUDA code.

## Prerequisites

| Optimized for            | Description
|:---                      |:---
| OS                       | Ubuntu* 22.04
| Hardware                 | Intel® Gen9 <br> Gen11 <br> Xeon CPU <br> Data Center GPU Max <br> Nvidia Testla P100 <br> Nvidia A100 <br> Nvidia H100 
| Software                 | SYCLomatic (Tag - 20230720) <br> Intel® oneAPI Base Toolkit (Base Kit) version 2023.2.1 <br> oneAPI for NVIDIA GPUs plugin (version 2023.2.0) from Codeplay

For more information on how to install Syclomatic Tool, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.v354cy) <br>
Refer [oneAPI for NVIDIA GPUs plugin](https://developer.codeplay.com/products/oneapi/nvidia/) from Codeplay to execute sample on NVIDIA GPU.

## Key Implementation Details

This sample demonstrates the migration of the following prominent CUDA features:
- Cooperative Groups
- Shared Memory

>  **Note**: Refer to [Workflow for a CUDA* to SYCL* Migration](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html#gs.s2njvh) for general information about the migration workflow.


### CUDA Source Code Evaluation

In this implementation, a random sequence of power of 2 elements is given as input, and the algorithm sorts the sequence in parallel. The algorithm sorts the first half of the list and the second half of the list separately. The algorithm then sorts the odd-indexed entries and the even-indexed entries separately. You need to make only one more comparison switch per pair of keys to sort the list completely.

In this sample, the array length of 1048576 is the input size for the algorithm. The code checks for all the input sizes in the intervals of 2nd power from array lengths from  64 to 1048576 calculated for one iteration. The comparator swaps the value if a top value is greater or equal to the bottom value.

> **Note**: For information on how to use SYCLomatic, refer to the materials at *[Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html)*.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Migrate the `Odd-Even Mergesort` Code

### Migrate the Code using SYCLomatic

For this sample, the SYCLomatic tool automatically migrates ~80% of the CUDA runtime APIs to SYCL. Follow these steps to generate the SYCL code using the compatibility tool.

1. Clone the required GitHub repository to your local environment.
   ```
   git clone https://github.com/NVIDIA/cuda-samples.git
   ```
2. Change to the odd-even mergesort sample directory.
   ```
   cd cuda-samples/Samples/2_Concepts_and_Techniques/sortingNetworks
   ```
3. Generate a compilation database with intercept-build
   ```
   intercept-build make
   ```
   The above step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.

4. Pass the JSON file as input to the SYCLomatic compatibility tool. The result is written to a folder named dpct_output. The --in-root specifies the path to the root of the source tree to be migrated. The `--gen-helper-function` option will make a copy of dpct header files/functions used in migrated code into the dpct_output folder as `include` folder.
   ```
   c2s -p compile_commands.json --in-root ../../.. --gen-helper-function
   ```

## Build and Run the `Odd-Even Mergesort` Sample

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
   
   **Note**: By default, no flag are enabled during build which supports Intel® UHD Graphics, Intel® Gen9, Gen11, Xeon CPU. <br>
    Enable **INTEL_MAX_GPU** flag during build which supports Intel® Data Center GPU Max 1550 or 1100 to get optimized performace. <br>
    Enable **NVIDIA_GPU** flag during build which supports NVIDIA GPUs.([oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/) plugin   from Codeplay is required to build for NVIDIA GPUs ) <br>
    
   By default, this command sequence will build the `02_sycl_migrated` versions of the program.

3. Run `02_sycl_migrated` for  GPU.
     ```
    $ make run
     ```
   Run `02_sycl_migrated` for  CPU.
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
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Example Output

The following example is for `02_sycl_migrated` for GPU on Intel(R) UHD Graphics P630 [0x3e96].

```
Running on Intel(R) UHD Graphics P630 [0x3e96]
Allocating and initializing host arrays...

Allocating and initializing CUDA arrays...

Running GPU oddevenMergesort (1 identical iterations)...

Testing array length 64 (16384 arrays per batch)...
Average time: 203.675995 ms


Validating the results...
...reading back GPU results
...inspecting keys array: OK
...inspecting keys and values array: OK
...stability property: NOT stable

Testing array length 128 (8192 arrays per batch)...
Average time: 5.088000 ms


Validating the results...
...reading back GPU results
...inspecting keys array: OK
...inspecting keys and values array: OK
...stability property: NOT stable

Testing array length 256 (4096 arrays per batch)...
Average time: 6.143000 ms


Validating the results...
...reading back GPU results
...inspecting keys array: OK
...inspecting keys and values array: OK
...stability property: NOT stable

Testing array length 512 (2048 arrays per batch)...
Average time: 7.210000 ms


Validating the results...
...reading back GPU results
...inspecting keys array: OK
...inspecting keys and values array: OK
...stability property: NOT stable

Testing array length 1024 (1024 arrays per batch)...
Average time: 11.907000 ms


Validating the results...
...reading back GPU results
...inspecting keys array: OK
...inspecting keys and values array: OK
...stability property: NOT stable

Testing array length 2048 (512 arrays per batch)...
Average time: 14.869000 ms


Validating the results...
...reading back GPU results
...inspecting keys array: OK
...inspecting keys and values array: OK
...stability property: NOT stable

Testing array length 4096 (256 arrays per batch)...
Average time: 18.344999 ms


Validating the results...
...reading back GPU results
...inspecting keys array: OK
...inspecting keys and values array: OK
...stability property: NOT stable

Testing array length 8192 (128 arrays per batch)...
Average time: 22.009001 ms


Validating the results...
...reading back GPU results
...inspecting keys array: OK
...inspecting keys and values array: OK
...stability property: NOT stable

Testing array length 16384 (64 arrays per batch)...
Average time: 26.112000 ms


Validating the results...
...reading back GPU results
...inspecting keys array: OK
...inspecting keys and values array: OK
...stability property: NOT stable

Testing array length 32768 (32 arrays per batch)...
Average time: 30.167999 ms


Validating the results...
...reading back GPU results
...inspecting keys array: OK
...inspecting keys and values array: OK
...stability property: NOT stable

Testing array length 65536 (16 arrays per batch)...
Average time: 34.814999 ms


Validating the results...
...reading back GPU results
...inspecting keys array: OK
...inspecting keys and values array: OK
...stability property: NOT stable

Testing array length 131072 (8 arrays per batch)...
Average time: 39.974998 ms


Validating the results...
...reading back GPU results
...inspecting keys array: OK
...inspecting keys and values array: OK
...stability property: NOT stable

Testing array length 262144 (4 arrays per batch)...
Average time: 45.018002 ms


Validating the results...
...reading back GPU results
...inspecting keys array: OK
...inspecting keys and values array: OK
...stability property: NOT stable

Testing array length 524288 (2 arrays per batch)...
Average time: 50.520000 ms


Validating the results...
...reading back GPU results
...inspecting keys array: OK
...inspecting keys and values array: OK
...stability property: NOT stable

Testing array length 1048576 (1 arrays per batch)...
Average time: 56.367001 ms

sortingNetworks-oddevenmergesort, Throughput = 18.6027 MElements/s, Time = 0.05637 s, Size = 1048576 elements, NumDevsUsed = 1, Workgroup = 256

Validating the results...
...reading back GPU results
...inspecting keys array: OK
...inspecting keys and values array: OK
...stability property: NOT stable

Shutting down...
Built target run

```
## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
 
