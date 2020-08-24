# `Merge Sort` Sample

The merge sort algorithm is a comparison-based sorting algorithm. In this sample, we use a top-down implementation, which recursively splits list into two halves (called sublists) until each sublist is of size 1. We then merge sublists two at a time in order to produce a sorted list. This sample could run in serial, or in parallel with OpenMP* Tasking #pragma omp task and #pragma omp taskwait.

For more details about merge sort algorithm and top-down implementation, please refer to http://en.wikipedia.org/wiki/Merge_sort.

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

Merge sort is a highly efficient recursive sorting algorithm. Known for its greater efficiency over other common sorting algorithms, it can compute in O(nlogn) time as opposed to O(n^2), making it a common choice for sorting implementations which deal with large quantities of elements. While it is already a very fast algorithm-- capable of sorting lists in a fraction of the time it would take an algorithm such as quick sort or insertion sort, it can be further accelerated with parallelism using OpenMP.

This code sample demonstrates how to convert a scalar implementation of merge sort into a parallelized version with minimal changes to the original, using OpenMP pragmas.


## Key Implementation Details 

The OpenMP* version of the merge sort implementation uses #pragma omp task in its recursive calls, which allows the recursive calls to be handled by different threads. The #pragma omp taskawait preceeding the function call to merge() ensures the two recursive calls are completed before merge() is executed. Through this use of OpenMP* pragmas, the recursive sorting algorithm can effectively run in parallel, where each recursion is a unique task able to be performed by any available thread.

 
## License  

This code sample is licensed under MIT license. 


## Building the `Merge Sort` Program

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


### Application Parameters 

There are two configurable options defined near the top of the code, both of which affect the program's performance:

constexpr int task_threshold - This determines the minimum size of the list passed to the OpenMP merge sort function required to recursively call itself and not the scalar version. Its purpose is to reduce the threading overhead as it gets less efficient on smaller list sizes. Settintg this value too small has the potential to reduce performance of the OpenMP implementation as it has more threading overhead for smaller workloads.

constexpr int n - This determines the size of the list used to test the merge sort functions. Setting it larger will result in a longer runtime, and is useful fpr analyzing the growth rate of the algorithm's runtime.


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