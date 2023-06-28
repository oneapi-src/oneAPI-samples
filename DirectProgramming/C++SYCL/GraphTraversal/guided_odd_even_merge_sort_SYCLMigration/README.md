# `odd-even mergesort` Sample

The `odd-even mergesort` sample demonstrates how to use the **odd-even mergesort** algorithm (also known as "Batcher's odd–even mergesort"), which belongs to the class of sorting networks. Generally, this algorithm is not efficient for large sequences compared to algorithms with better asymptotic algorithmic complexity (merge sort or radix sort); however, this sort method might be the preferred algorithm for sorting batches of short-sized to mid-sized (key, value) array pairs.

| Area                | Description
|:---                 |:---
| What you will learn | How to begin migrating CUDA to SYCL
| Time to complete    | 15 minutes

## Purpose

The **odd-even mergesort** algorithm was developed by Kenneth Edward Batcher.

The algorithm is based on a merge algorithm that merges two sorted halves of a sequence into a sorted sequence. In contrast to a standard merge sort, this algorithm is not data-dependent. All comparisons are performed regardless of the actual data, so you can implement this algorithm as a sorting network.

> **Note**: We use Intel® open-sources SYCLomatic tool, which assists developers in porting CUDA code automatically to SYCL code. To finish the process, developers complete the rest of the coding manually and then tune to the desired level of performance for the target architecture. You can use SYCLomatic tool that comes along with the Intel® oneAPI Base Toolkit.

## Prerequisites

| Property             | Description
|:---                  |:---
| OS                   | Ubuntu* 20.04
| Hardware             | Intel® Gen9 (and newer) <br> Intel® Xeon® Processors
| Software             | SYCLomatic version 2023.1 <br> Intel® oneAPI Base Toolkit (Base Kit)

For more information on how to install SYCLomatic Tool, refer to the [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.v3584e) page.

> **Note**: Read the [Get Started with the Intel® oneAPI Base Toolkit for Linux*](https://www.intel.com/content/www/us/en/docs/oneapi-base-toolkit/get-started-guide-linux/2023-1/overview.html) document for information on how to start using the Intel® oneAPI Base Toolkit.

## Key Implementation Details

This sample demonstrates the migration of the following prominent CUDA features:
- Cooperative Groups
- Shared Memory

This sample contains two versions of the program.

| Folder Name             | Description
|:---                     |:---
| `01_dpct_output`        | Contains output of SYCLomatic Tool used to migrate SYCL-compliant code from CUDA code. This SYCL code has some unmigrated code that has to be manually fixed to get full functionality.
| `02_sycl_migrated`      | Contains manually migrated SYCL code from CUDA code.

### Workflow for CUDA to SYCL Migration

Refer to the [Workflow for a CUDA* to SYCL* Migration](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html#gs.s2njvh) for details.

### CUDA Source Code Evaluation

In this implementation, a random sequence of power of 2 elements is given as input, and the algorithm sorts the sequence in parallel. The algorithm sorts the first half of the list and second half of the list separately. The algorithm then sorts the odd-indexed entries and the even-indexed entries separately. You need make only one more comparison-switch per pair of keys to sort the list completely.

In this sample, the array length of 1048576 is the input size for the algorithm. The code checks for all the input sizes in the intervals of 2th power from array lengths from  64 to 1048576 calculated for one iteration. The comparator swaps the value if top value is greater or equal to the bottom value.

This sample is migrated from NVIDIA CUDA sample. See the [sortingNetworks](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/sortingNetworks) sample in the NVIDIA/cuda-samples GitHub.

## Build and Run the `odd-even mergesort` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*.


## Assisted Migration with SYCLomatic

For this sample, the SYCLomatic Tool automatically migrates 100% of the CUDA code to SYCL. Follow these steps to generate the SYCL code using the compatibility tool:

1. Clone the repository.
   ```
   git clone https://github.com/NVIDIA/cuda-samples.git
   ```
2. Change to the relevant folder.
   ```
   cd cuda-samples/Samples/2_Concepts_and_Techniques/sortingNetworks
   ```
3. Generate a compilation database with intercept-build
   ```
   intercept-build make
   ```
   This step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.

5. Pass the JSON file as input to the SYCLomatic compatibility tool. The result is written to a folder named dpct_output. The --in-root specifies path to the root of the source tree to be migrated.
   ```
   c2s -p compile_commands.json --in-root ../../.. --use-custom-helper=api
   ```

### On Linux*

1. Change to the sample directory.

2. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make
   ```
   By default, this command sequence will build the `01_dpct_output`and `02_sycl_migrated` versions of the program.

3. Run `02_sycl_migrated` for  GPU.
     ```
    make run
     ```
   Run `02_sycl_migrated` for  CPU.
     ```
    export ONEAPI_DEVICE_SELECTOR=cpu
    make run
    ```

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
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
 
