# `oneDNN Simple Model` Sample

This sample is implemented in C++ and DPC++ and runs on CPU or GPU. The sample
also includes [Jupyer notebook](https://github.com/oneapi-src/oneAPI-samples/blob/master/Libraries/oneDNN/tutorials/tutorial_simple_model.ipynb) that
demonstrates how to port an oneDNN code sample from CPU-only version to CPU & GPU
in Intel® DevCloud for oneAPI environment.

| Optimized for                      | Description
| :---                               | :---
| OS                                 | Linux* Ubuntu* 18.04; Windows 10
| Hardware                           | Skylake with GEN9 or newer
| Software                           | oneAPI Deep Neural Network Library (oneDNN), oneAPI DPC++/C++ Compiler, oneAPI Threading Building Blocks (oneTBB), GNU Compiler Collection, Intel® C++ Compiler
| What you will learn                | Running a simple convolutional model on Intel CPU or Intel GPU
| Time to complete                   | 15 minutes

## Purpose

This sample implements the computational part of a convolutional neural network
based on [ImageNet Classification with Deep Convolutional Neural Networks by Alex Krizhevsky et al.](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
The network consists of 15 layers, including convolution, rectified linear
unit (ReLU), linear response normalization (LRN), and inner product.

With this sample, you will learn:
* How to run a simple convolutional network on Intel CPU or Intel GPU
* How to compile examples with oneAPI DPC++/C++ Compiler, Intel® C++ Compiler,
and GNU C++ Compiler
* How to switch between OpenMP and TBB for CPU parallelization
* How to describe tensors with oneDNN memory objects
* How to describe neural network layers with oneDNN primitives

The sample executes on the system's CPU by default and can be executed on Intel GPU
using a command line parameter `gpu`.

## Key Implementation Details

This sample uses example file `${DNNLROOT}/examples/cnn_inference_fp32.cpp`
from oneDNN distribution. You can find this code in
[oneDNN Github repository](https://github.com/oneapi-src/oneDNN/blob/dev-v2/examples/cnn_inference_f32.cpp).

Detailed code walkthrough is available in [oneDNN developer guide](https://oneapi-src.github.io/oneDNN/v2/cnn_inference_f32_cpp.html)

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

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
>NOTE: The source file "cnn_inference_f32.cpp" will be copied from ${INTEL_ONEAPI_INSTALL_FOLDER}/dnnl/latest/cpu_dpcpp_gpu_dpcpp/examples/ to build/src folder. Users can rebuild the cnn_inference_f32.cpp by typing "make" under build folder.
3. Run the program
```
./bin/cnn-inference-f32-cpp
```

By default, the sample uses oneAPI DPC++/C++ Compiler and can execute on CPUs or
Intel GPUs. You can build the sample with CPU support with other compilers
and threading runtimes:
* GNU C++ Compiler and GNU OpenMP runtime
```
source ${INTEL_ONEAPI_INSTALL_FOLDER}/setvars.sh --dnnl-configuration=cpu_gomp
CC=GCC CXX=g++ cmake ..
```
* Intel® C++ Compiler and Intel OpenMP runtime
```
source ${INTEL_ONEAPI_INSTALL_FOLDER}/setvars.sh --dnnl-configuration=cpu_iomp
CC=icc CXX=icpc cmake ..
```
* Intel® C++ Compiler and TBB runtime
```
source ${INTEL_ONEAPI_INSTALL_FOLDER}/setvars.sh --dnnl-configuration=cpu_tbb
CC=icc CXX=icpc cmake ..
```

### On a Windows* System

Open "Intel oneAPI command prompt for Intel 64 for Visual Studio 2017" or
"Intel oneAPI command prompt for Intel 64 for Visual Studio 2019" and perform the following steps:
1. Setup oneAPI development environment
```
C:\Program Files (x86)\Intel\oneAPI\setvars.bat
```
2. Build the program using `cmake`
```
cd C:\Program Files (x86)\Intel\oneAPI\dnnl\latest\cpu_dpcpp_gpu_dpcpp\examples\
mkdir build
cd build
set CC=clang
set CXX=clang++
cmake -G Ninja .. -DDNNL_CPU_RUNTIME=DPCPP -DDNNL_GPU_RUNTIME=DPCPP
cmake --build .
```

### Include Files
The include folder is located at ${DNNLROOT}\include on your development system".

3. Run the program
```
cnn-inference-f32-cpp.exe
```

### Include Files

The include folder is located at ${DNNLROOT}\include on your development system".

## Running the Sample

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

### Application Parameters

You can specify the target device for this sample using command-line arguments:
* `cpu` (default) directs the application to run on the system's CPU
* `gpu` directs the sample to run on Intel GPU

> Note: When executed with `gpu` parameter the
> sample will return an error if there are no Intel GPUs are found in the system.

You can get additional information during the execution of this sample by setting
environment variable `DNNL_VERBOSE=1`.

#### On a Linux System
```
export DNNL_VERBOSE=1
```
#### On a Windows* System
```
set DNNL_VERBOSE=1
```

### Example of Output

```
Use time: 28.84 ms per iteration.
Example passed on CPU.
```

When executed with `DNNL_VERBOSE=1`:
```
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

