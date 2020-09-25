# Histogram Sample

This sample demonstrates a histogram that groups numbers together and provides the count of a particular number in the input. In this sample we are using dpstd APIs to offload the computation to the selected device.

For comprehensive instructions regarding DPC++ Programming, go to https://software.intel.com/en-us/oneapi-programming-guide and search based on relevant terms noted in the comments.

| Optimized for                   | Description                                                                                          |
|---------------------------------|------------------------------------------------------------------------------------------------------|
| OS                              | Linux Ubuntu 18.04                                                                                   |
| Hardware                        | Skylake with GEN9 or newer, Intel(R) Programmable Acceleration Card with Intel(R) Arria(R) 10 GX FPGA|
| Software                        | Intel® oneAPI DPC++ Compiler (beta)                                                                  |


## Purpose
This sample creates both dense and sparse histograms using dpstd APIs, on an input array of 1000 elements with values chosen randomly berween 0 and 9. To differentiate between sparse and dense histogram, we make sure that one of the values never occurs in the input array, i.e. one bin will have always have 0 value.

For the dense histogram all the bins(including the zero-size bins) are stored, whereas for the sparse algorithm only non-zero sized bins are stored.

The computations are performed using Intel® oneAPI DPC++ library (oneDPL).

## Key Implementation Details
The basic DPC++ implementation explained in the code includes accessor,
kernels, queues, buffers as well as some oneDPL library calls.

## License

This code sample is licensed under MIT license.

## Building the histogram program for CPU and GPU
On a Linux* System
Perform the following steps:

```
mkdir build
cd build
cmake ..
```

Build the program using the following make commands
```
make
```

Run the program using:
```
make run or src/histogram
```

Clean the program using:
```
make clean
```
If you see the following error message when compiling this sample:

```
Error 'dpc_common.hpp' file not found
```
You need to add the following directory to the list of include folders, that are required by your project, in your project's Visual Studio project property panel. The missing include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

## Known issues

The sample is prone to ``Floating point exception`` and `CL_INVALID_WORK_GROUP_SIZE` errors on GPU with DPC++ L0 backend, which can be avoided
by setting `_PSTL_COMPILE_KERNEL` macro to `0`. You can do it using the following command before running `cmake`:

```
export CXXFLAGS=-D_PSTL_COMPILE_KERNEL=0
```

## Running the Sample

Application Parameters
You can modify the histogram from within src/main.cpp. The functions sparse_histogram() and dense_histogram() can be reused for any set of input values.

#### Example of Output

Input:
1 1 8 1 8 6 1 0 1 5 5 2 2 8 1 2 1 1 1 6 2 1 1 8 3 6 6 2 2 1 1 8 1 0 0 0 2 2 7 6 5 1 6 1 1 6 1 5 1 0 0 1 1 1 0 5 5 0 7 0 1 6 0 5 7 0 3 0 0 0 0 6 0 2 5 5 6 6 8 7 6 6 8 8 7 7 2 2 0 7 2 2 5 2 7 1 3 0 1 1 0 1 7 2 0 1 5 1 7 0 8 3 1 5 0 6 1 0 8 2 7 2 1 1 1 3 2 5 1 2 5 1 6 3 3 1 3 8 0 1 1 8 2 0 2 0 1 2 0 2 1 8 1 6 0 6 7 1 1 8 3 6 0 7 7 1 6 1 7 6 1 8 3 3 6 3 1 2 7 2 1 0 1 8 7 0 5 5 1 1 3 2 1 3 7 0 3 2 1 1 8 0 1 0 2 5 3 6 7 0 6 2 0 8 8 5 6 3 0 5 7 3 5 0 0 3 7 7 5 6 7 2 7 8 0 0 2 3 0 1 3 1 1 2 7 1 5 1 0 3 7 2 0 3 0 0 6 7 5 0 5 3 0 3 0 0 1 3 2 5 2 3 6 3 5 5 2 0 7 6 3 6 7 6 0 7 6 5 6 0 3 0 2 1 1 0 2 2 1 1 7 3 8 2 5 2 7 7 2 1 3 2 1 1 1 8 6 5 2 3 3 6 1 5 8 2 1 1 2 5 2 0 7 3 3 3 3 8 8 0 1 2 8 2 3 7 0 8 1 2 2 1 6 2 8 5 1 3 5 7 8 0 5 2 1 8 7 0 6 7 8 7 7 5 8 0 3 8 8 2 8 1 7 2 1 6 0 0 7 3 2 2 1 7 0 2 5 7 5 2 3 1 0 2 1 6 2 2 3 1 5 3 0 3 5 0 7 3 1 5 7 6 7 8 2 7 0 7 2 5 7 5 0 6 5 8 3 7 0 7 6 5 8 5 6 2 5 2 5 0 5 1 1 3 1 6 0 8 3 0 0 1 7 2 5 2 0 7 2 0 3 7 3 0 3 0 2 6 0 7 6 5 0 1 8 8 5 8 7 8 1 0 8 0 2 2 2 2 0 2 0 3 0 3 3 3 3 3 7 3 2 0 6 0 3 0 8 0 1 1 6 3 1 3 1 0 6 3 7 1 5 7 8 6 0 0 7 1 1 6 3 2 8 0 2 3 0 1 1 6 3 5 7 7 0 8 2 1 0 7 8 5 2 5 0 0 6 6 5 8 3 8 1 2 7 5 3 2 1 0 8 7 8 1 3 8 1 3 3 1 2 0 5 1 6 3 6 1 0 2 7 3 0 8 1 7 2 5 7 6 8 5 2 7 0 5 6 2 8 7 1 8 7 2 3 2 8 0 3 8 1 1 1 1 7 5 6 0 8 2 6 7 7 8 5 8 2 2 8 2 7 0 1 6 3 5 8 2 3 1 1 2 0 2 3 8 5 7 8 5 1 1 1 8 1 7 5 0 7 1 0 6 3 5 1 6 8 0 6 1 8 7 5 0 8 7 6 2 5 5 5 6 7 7 1 0 5 0 2 3 3 6 0 1 0 1 8 7 0 5 8 6 3 2 2 0 0 1 3 6 5 8 1 3 2 5 1 0 6 3 0 7 7 2 2 8 2 1 1 2 6 3 6 7 5 2 8 6 3 0 1 8 6 0 1 2 6 0 0 1 2 2 8 0 5 1 6 7 0 1 7 6 1 2 2 8 6 8 5 8 8 1 5 1 1 6 6 8 7 6 0 0 0 6 7 3 5 5 8 5 2 6 2 7 8 3 6 1 2 0 1 2 1 6 6 6 2 1 6 7 5 0 5 3 2 3 6 7 6 5 2 2 0 1 0 7 7 6 0 8 1 1 1 8 7 5 3 7 1 0 5 0 3 1 2 5 5 8 1 0 3 5 0 1 8 0 6 0 0 6 3 8 5 2 5 1 5 0 2 0 7 6 8 1 7 1 0 1 0 6 0 1 0 0 1 8 1 7 2 3 3 5 1 8 6 6 1 2 2 2 3 1 8 2 2 6 3 7 6 1 2 6 1 2 6 2 0 5 0 2 7 3 5 8 3 2 3 1 5 6 6 6 7 3 8 0 8 0 5 5 8 5 0 0 6 2 0 6 8 1 6 6 2 0 3 5 3 2 8 6 1 3 3 8 7 0 7 6 7 1 0 6 7 0 5 0 0 5 8 1
Dense Histogram:
[(0, 161) (1, 170) (2, 136) (3, 108) (4, 0) (5, 105) (6, 110) (7, 108) (8, 102) ]
Sparse Histogram:
[(0, 161) (1, 170) (2, 136) (3, 108) (5, 105) (6, 110) (7, 108) (8, 102) ]

