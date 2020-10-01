# oneDNN Simple Model Sample

This sample is implemented in C++ and DPC++ and runs on CPU or GPU. The sample
also includes [a Jupyter Notebook](simple_model.ipynb) that
demonstrates how to port a oneDNN sample from CPU-only version to CPU & GPU
in Intel oneAPI DevCloud environment.

| Optimized for                      | Description
| :---                               | :---
| OS                                 | Linux* Ubuntu* 18.04; Windows 10
| Hardware                           | Skylake with GEN9 or newer
| Software                           | Intel oneAPI Deep Neural Network Library (oneDNN), Intel oneAPI DPC++/C++ Compiler, Intel oneAPI Threading Building Blocks (oneTBB), GNU Compiler Collection, Intel C++ Compiler
| What you will learn                | Running a simple convolutional model on Intel CPU or Intel GPU
| Time to complete                   | 15 minutes

## Purpose

This sample implements computational part of a convolutional neural network
based on [ImageNet Classification with Deep Convolutional Neural Networks by Alex Krizhevsky at al](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
The network consists of 15 layers including convolution, rectified linear
unit (ReLU), linear response normalization (LRN), and inner product.

With this sample you will learn:
* How to run a simple convolutional network on Intel CPU or Intel GPU
* How to compile examples with Intel oneAPI DPC++/C++ Compiler, Intel C++ Compiler,
and GNU C++ Compiler
* How to switch between OpenMP and TBB for CPU parallelization
* How to describe tensors with oneDNN memory objects
* How to describe neural network layers with oneDNN primitives

The sample executes on system's CPU by default and can be executed on Intel GPU
using a command line parameter `gpu`.

## Key Implementation Details

This sample uses example file `${DNNLROOT}/examples/cnn_inference_fp32.cpp`
from oneDNN distribution. You can find this code in
[oneDNN Github repository](https://github.com/oneapi-src/oneDNN/blob/dev-v2/examples/cnn_inference_f32.cpp).

Detailed code walkthrough is available in [oneDNN developer guide](https://oneapi-src.github.io/oneDNN/v2/cnn_inference_f32_cpp.html)

## License

This code sample is licensed under MIT license.

## Building the sample for CPU and GPU

### On a Linux System

Perform the following steps:
1. Setup oneAPI development environment
```
source ${INTEL_ONEAPI_INSTALL_FOLDER}/setvars.sh
```
2. Build the program using `cmake`
```
mkdir build
cd build
cmake ..
make
```
3. Run the program
```
./bin/cnn-inference-f32-cpp
```

By default the sample uses oneAPI DPC++/C++ Compiler and can execute on CPUs or
Intel GPUs. You can build the sample with CPU support with other compilers
and threading runtimes:
* GNU C++ Compiler and GNU OpenMP runtime
```
source ${INTEL_ONEAPI_INSTALL_FOLDER}/setvars.sh --dnnl-configuration=cpu_gomp
CC=GCC CXX=g++ cmake ..
```
* Intel C++ Compiler and Intel OpenMP runtime
```
source ${INTEL_ONEAPI_INSTALL_FOLDER}/setvars.sh --dnnl-configuration=cpu_iomp
CC=icc CXX=icpc cmake ..
```
* Intel C++ Compiler and TBB runtime
```
source ${INTEL_ONEAPI_INSTALL_FOLDER}/setvars.sh --dnnl-configuration=cpu_tbb
CC=icc CXX=icpc cmake ..
```

### On a Windows* System Using Visual Studio* Version 2017 or Newer

Open "x64 Native Tools Command Prompt for VS2017" or 
"x64 Native Tools Command Prompt for VS2019" and perform the following steps:
1. Setup oneAPI development environment
```
C:\Program Files (x86)\intel\oneapi\setvars.bat --dnnl-configuration=cpu_vcomp
```
2. Build the program using `cmake`
```
mkdir build
cd build
cmake -G "Visual Studio 16 2019" ..
cmake --build .
```

> Note: You can open the `simple_model.sln` in build folder to edit source
> code with Microsoft Visual Studio integrated development environment.

### Include Files
The include folder is located at ${DNNLROOT}\include on your development system".

3. Run the program
```
bin\Debug\cnn-inference-f32-cpp.exe
```

### Include Files

The include folder is located at ${DNNLROOT}\include on your development system".

## Running the Sample

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### Application Parameters

You can specify target device for this sample using command line arguments:
* `cpu` (default) directs the application to run on system's CPU
* `gpu` directs the sample to run on Intel GPU

> Note: When executed with `gpu` parameter the 
> sample will return an error if there are no Intel GPUs are found in the system.

You can get additional information during execution of this sample by setting
environment variable `DNNL_VERBOSE=1`.

### Example of Output

```
Use time: 28.84 ms per iteration.
Example passed on CPU.
```

When executed with `DNNL_VERBOSE=1`:
```
dnnl_verbose,info,oneDNN v1.95.0 (commit ae08a30fff7f76759fd4c5093c01707d0ee12c4c)
dnnl_verbose,info,cpu,runtime:DPC++
dnnl_verbose,info,cpu,isa:Intel AVX2
dnnl_verbose,info,gpu,runtime:DPC++
dnnl_verbose,info,cpu,engine,0,backend:OpenCL,name:Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz,driver_version:2020.10.7
dnnl_verbose,info,gpu,engine,0,backend:Level Zero,name:Intel(R) Gen12LP,driver_version:0.8.0
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:Acdb8a:f0,,,96x3x11x11,0.24292
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcde:f0 dst_f32::blocked:aBCde8c8b:f0,,,2x128x48x5x5,0.26709
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:ABcd8b8a:f0,,,384x256x3x3,1.16699
...
Use time: 20.11 ms per iteration.
Example passed on CPU.
```

