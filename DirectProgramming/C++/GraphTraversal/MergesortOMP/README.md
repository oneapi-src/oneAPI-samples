# `MergeSort OMP` Sample

The `MergeSort OMP` sample uses merge sort, which is a comparison-based sorting algorithm. In this sample, we use a top-down implementation, which recursively splits the list into two halves (called sublists) until each sublist is size 1.

>**Note**: For more details, see the [Merge sort](http://en.wikipedia.org/wiki/Merge_sort) article on the algorithm and top-down implementation.

| Area                  | Description
|:---                   |:---
| What you will learn   | How to accelerate a scalar program using OpenMP* tasks
| Time to complete      | 15 minutes


## Purpose

Merge sort is a highly efficient recursive sorting algorithm. Known for its greater efficiency over other common sorting algorithms, it can compute in O(nlogn) time instead of O(n^2), making it a common choice for sorting implementations that deal with large quantities of elements. While it is a very fast algorithm and capable of sorting lists faster than other algorithms, like quicksort or insertion sort, you can accelerate merge sort more with parallelism using OpenMP.

We then merge sublists two at a time to produce a sorted list. This sample could run in serial or parallel with OpenMP* Tasking `#pragma omp task` and `#pragma omp taskwait`.

## Prerequisites

| Optimized for          | Description
|:---                    |:---
| OS                     | Ubuntu* 18.04
| Hardware               | Skylake with GEN9 or newer
| Software               | Intel® oneAPI DPC++ Compiler

## Key Implementation Details

This code sample demonstrates how to convert a scalar implementation of merge sort into a parallelized version with minimal changes to the original, using OpenMP pragmas.

The OpenMP* version of the merge sort implementation uses the `#pragma omp task` in its recursive calls, which allows the recursive calls to be handled by different threads. The `#pragma omp taskawait` preceding the function call to `merge()` ensures the two recursive calls complete before the `merge()` is executed. Through this use of OpenMP* pragmas, the recursive sorting algorithm can effectively run in parallel, where each recursion is a unique task able to be performed by any available thread.

Performance number tabulation.

| Version            | Performance Data
|:---                |:---
| Scalar baseline    | 1.0
| OpenMP* Task       | 4.0x speedup


## Build the `MergeSort OMP` Program

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or Linux*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### Use Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the 
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*

1. Build the program
   ```
   make
   ```
   Alternatively, you can enable the performance tabulation mode then build the program.

   ```
   export perf_num=1
   make
   ```

#### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Run the `MergeSort OMP` Program

### Configurable Parameters

There are two configurable options defined in the source code. Both parameters affect program performance.

- `constexpr int task_threshold` - This determines the minimum size of the list passed to the OpenMP merge sort function required to call itself and not the scalar version recursively. Its purpose is to reduce the threading overhead as it gets less efficient on smaller list sizes. Setting this value too small can reduce the OpenMP implementation's performance as it has more threading overhead for smaller workloads.
- `constexpr int n` - This determines the size of the list used to test the merge sort functions. Setting it larger will result in longer runtime and is useful for analyzing the algorithm's runtime growth rate.

### On Linux

1. Run the program.
   ```
   make run
   ```

2. Clean the program. (Optional)
   ```
   make clean
   ```

## Example Output

You are prompted to select a test type. The following example output shows the results of choosing `[0] all tests`.

```
N = 100000000
Merge Sort Sample
[0] all tests
[1] serial
[2] OpenMP Task
0

Running all tests

Serial version:
Shuffling the array
Sorting
Sort succeeded in 11.9732 seconds.

OpenMP Task Version:
Shuffling the array
Sorting
Sort succeeded in 3.17086 seconds.
```

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
