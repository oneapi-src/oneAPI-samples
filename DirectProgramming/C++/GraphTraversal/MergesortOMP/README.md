# `Merge Sort` Sample

The merge sort algorithm is a comparison-based sorting algorithm. In this sample, we use a top-down implementation, which recursively splits the list into two halves (called sublists) until each sublist is size 1. We then merge sublists two at a time to produce a sorted list. This sample could run in serial or parallel with OpenMP* Tasking #pragma omp task and #pragma omp taskwait.

For more details, see the wiki on [merge sort](http://en.wikipedia.org/wiki/Merge_sort) algorithm and top-down implementation.

| Optimized for                     | Description
|:---                               |:---
| OS                                | MacOS Catalina or newer
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI C++ Compiler Classic
| What you will learn               | How to accelerate a scalar program using OpenMP* tasks
| Time to complete                  | 15 minutes

Performance number tabulation

| Version                           | Performance data
|:---                               |:---
| Scalar baseline                   | 1.0
| OpenMP Task                       | 4.0x speedup


## Purpose

Merge sort is a highly efficient recursive sorting algorithm. Known for its
greater efficiency over other common sorting algorithms, it can compute in
O(nlogn) time instead of O(n^2), making it a common choice for sorting
implementations that deal with large quantities of elements. While it is
already a very fast algorithm-- capable of sorting lists in a fraction of the
time it would take an algorithm such as quicksort or insertion sort, it can be
further accelerated with parallelism using OpenMP.

This code sample demonstrates how to convert a scalar implementation of merge
sort into a parallelized version with minimal changes to the original, using
OpenMP pragmas.


## Key Implementation Details

The OpenMP* version of the merge sort implementation uses the #pragma omp task
in its recursive calls, which allows the recursive calls to be handled by
different threads. The #pragma omp taskawait preceding the function call to
merge() ensures the two recursive calls are completed before the merge() is
executed. Through this use of OpenMP* pragmas, the recursive sorting algorithm
can effectively run in parallel, where each recursion is a unique task able to
be performed by any available thread.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)



### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

## Building the `Merge Sort` Program

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

Perform the following steps:
1. Build the program using the following `make` commands.
```
$ export perf_num=1     *optional, will enable performance tabulation mode
$ make
```

## Running the Sample

2. Run the program:
    ```
    make run
    ```

3. Clean the program using:
    ```
    make clean
    ```

If an error occurs, troubleshoot the problem using the Diagnostics Utility for
Intel® oneAPI Toolkits.
[Learn more](https://software.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)

### Application Parameters

There are two configurable options defined near the top of the code, both of
which affect the program's performance:

- constexpr int task_threshold - This determines the minimum size of the list passed to the OpenMP merge sort function required to call itself and not the scalar version recursively. Its purpose is to reduce the threading overhead as it gets less efficient on smaller list sizes. Setting this value too small can reduce the OpenMP implementation's performance as it has more threading overhead for smaller workloads.
- constexpr int n - This determines the size of the list used to test the merge sort functions. Setting it larger will result in longer runtime and is useful for analyzing the algorithm's runtime growth rate.


### Example of Output
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
