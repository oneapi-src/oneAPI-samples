# Stable sort by key sample

Stable sort by key is a sorting operation when sorting of 2 sequences (keys and values) only keys are compared but both keys and values are swapped. This sample demonstrates `counting_iterator` and `zip_iterator` from Intel&reg; oneAPI DPC++ Library (oneDPL).


| Optimized for                   | Description                                                                      |
|---------------------------------|----------------------------------------------------------------------------------|
| OS                              | Linux* Ubuntu* 18.04                                                             |
| Hardware                        | Skylake with GEN9 or newer                                                       |
| Software                        | Intel&reg; oneAPI DPC++/C++ Compiler; Intel&reg; oneAPI DPC++ Library (oneDPL)   |
| What you will learn             | How to use `counting_iterator` and `zip_iterator`                                |
| Time to complete                | At most 5 minutes                                                                |

## Purpose

The sample models stable sorting by key: during the sorting of 2 sequences (keys and values) only keys are compared but both keys and values are swapped.
It fills two buffers (one of the buffer is filled using `counting_iterator`) and then sorts them both using `zip_iterator`.

The sample demonstrates how to use `counting_iterator` and `zip_iterator` using Intel&reg; oneAPI DPC++ library (oneDPL).
* `counting_iterator` helps to fill the sequence with the numbers zero through `n` using std::copy.
* `zip_iterator` provides the ability to iterate over several sequences simultaneously.

## Key Implementation Details

Following Parallel STL algorithms are used in the code: `transform`, `copy`, `stable_sort`.

`counting_iterator`, `zip_iterator` are used from Extension API of oneDPL.

## License

This code sample is licensed under MIT license.

## Building the 'Stable sort by key' Program for CPU and GPU

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel(R) oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
Perform the following steps:

1. Build the program using the following `cmake` commands.
```
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
```

2. Run the program:
```
    $ make run
```

3. Clean the program using:
```
    $ make clean
```

## Running the Sample
### Example of Output

```
success
Run on Intel(R) Gen9
```
