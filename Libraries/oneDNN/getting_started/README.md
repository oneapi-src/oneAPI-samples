# `oneDNN Getting Started` Sample

oneAPI Deep Neural Network Library (oneDNN) is an open-source performance
library for deep learning applications. The library includes basic building
blocks for neural networks optimized for Intel Architecture Processors
and Intel Processor Graphics. oneDNN is intended for deep learning
applications and framework developers interested in improving application
performance on Intel CPUs and GPUs.
You can find library source code and code used by these samples at the oneDNN Github repository.

This sample guides users how to compile oneDNN applications and executes the binaries on different Intel architectures. The sample also
also includes [a Jupyer Notebook](https://github.com/oneapi-src/oneAPI-samples/blob/master/Libraries/oneDNN/tutorials/tutorial_getting_started.ipynb) that
demonstrates how to compile the code with various oneDNN configurations
in Intel® DevCloud for oneAPI environment.

| Optimized for                      | Description
| :---                               | :---
| OS                                 | Linux* Ubuntu* 18.04;
| Hardware                           | Skylake with GEN9 or newer
| Software                           | oneAPI Deep Neural Network Library (oneDNN), oneAPI DPC++/C++ Compiler, oneAPI Threading Building Blocks (oneTBB), GNU Compiler Collection, Intel® C++ Compiler
| What you will learn                | Running a simple convolutional model on Intel CPU or Intel GPU
| Time to complete                   | 15 minutes

## Purpose

This sample demonstrates how to compile oneDNN applications and executes the binaries on different Intel architectures by using Intel® oneAPI Base Toolkit.  
With this code sample, you will learn:
* How to configure oneDNN environment for different pre-built oneDNN binaries from Intel® oneAPI Base Toolkit.
* How to compile oneDNN applications with different pre-built oneDNN binaries from Intel® oneAPI Base Toolkit.
* How to run oneDNN applications on different Intel architectures.
* How to enable oneDNN verbose log for validation.

All compiled binaries execute on the system's CPU by default and can be executed on Intel GPU
using a command line parameter `gpu`.

## Key Implementation Details

This sample uses example files under `${DNNLROOT}/examples/`
from oneDNN distribution. You can find those codes in
[oneDNN Github repository](https://github.com/oneapi-src/oneDNN/blob/dev-v2/examples/).

Detailed code walkthrough is available in [oneDNN developer guide](https://oneapi-src.github.io/oneDNN/v2/dev_guide_examples.html)




## Run the Sample

### On a Linux System

Perform the following steps:
#### 1. Setup oneAPI development environment
```
source ${INTEL_ONEAPI_INSTALL_FOLDER}/setvars.sh
```
#### 2. Build the program using `cmake`
```
mkdir build
cd build
cmake ..
make
```
> NOTE: The source files such as "getting_started.cpp" will be copied from  
> ${INTEL_ONEAPI_INSTALL_FOLDER}/dnnl/latest/cpu_dpcpp_gpu_dpcpp/examples/  
> to build/src folder. Users can rebuild the all source codes by typing  
> "make" under build folder.

#### 3. Run the sample
You can get additional information during the execution of this sample by setting
environment variable `DNNL_VERBOSE=1`.

Take getting_started.cpp as an example.
```
export DNNL_VERBOSE=1
```
```
./bin/getting-started-cpp
```

#### Using different pre-built oneDNN binaries from Intel® oneAPI Base Toolkit 

By default, the sample uses oneAPI DPC++/C++ Compiler and can execute on CPUs or
Intel GPUs. You can build the sample with CPU support with other compilers
and threading runtimes:
* GNU C++ Compiler and GNU OpenMP runtime
```
source ${INTEL_ONEAPI_INSTALL_FOLDER}/setvars.sh --dnnl-configuration=cpu_gomp  
mkdir build  
cd build  
CC=GCC CXX=g++ cmake ..  
make  
```
* Intel® C++ Compiler and Intel OpenMP runtime
```
source ${INTEL_ONEAPI_INSTALL_FOLDER}/setvars.sh --dnnl-configuration=cpu_iomp  
mkdir build  
cd build  
CC=icc CXX=icpc cmake ..  
make  
```
* Intel® C++ Compiler and TBB runtime
```
source ${INTEL_ONEAPI_INSTALL_FOLDER}/setvars.sh --dnnl-configuration=cpu_tbb  
mkdir build  
cd build  
CC=icc CXX=icpc cmake ..  
make  
```
### On a Windows* System

Open "Intel oneAPI command prompt for Intel 64 for Visual Studio 2017" or
"Intel oneAPI command prompt for Intel 64 for Visual Studio 2019" and perform the following steps:

#### Microsoft Visual Studio Compiler

##### 1. Setup oneAPI development environment
```
C:\Program Files (x86)\Intel\oneAPI\setvars.bat --dnnl-configuration=cpu_vcomp  --force
```
##### 2. Build the program using `cmake`
```
mkdir build
cd build
cmake -G "Visual Studio 16 2019" ..
cmake --build .
```
##### 3. Run the sample
You can get additional information during the execution of this sample by setting
environment variable `DNNL_VERBOSE=1`.

```
set DNNL_VERBOSE=1
```

Take getting_started.cpp as an example.

```
bin\Debug\getting-started-cpp.exe
```

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### Application Parameters

You can specify the target device for this sample using command-line arguments:
* `cpu` (default) directs the application to run on the system's CPU
* `gpu` directs the sample to run on Intel GPU

> Note: When executed with `gpu` parameter the
> sample will return an error if the sample is compiled with oneDNN configuration
> that does not support GPU, or no Intel GPUs are found in the system.

You can get additional information during the execution of this sample by setting
environment variable `DNNL_VERBOSE=1`.  

### Include Files

The include folder is located at ${DNNLROOT}\include on your development system".


## Example of Output

```
Example passed on CPU.
```

When executed with `DNNL_VERBOSE=1`:
```
dnnl_verbose,info,cpu,runtime:DPC++
dnnl_verbose,info,cpu,isa:Intel AVX2
dnnl_verbose,info,gpu,runtime:DPC++
dnnl_verbose,info,cpu,engine,0,backend:OpenCL,name:Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz,driver_version:2020.10.7
dnnl_verbose,info,gpu,engine,0,backend:Level Zero,name:Intel(R) Gen12LP,driver_version:0.8.0
dnnl_verbose,exec,cpu,eltwise,jit:avx2,forward_inference,data_f32::blocked:acdb:f0 diff_undef::undef::f0,,alg:eltwise_relu alpha:0 beta:0,1x3x13x13,0.125
Example passed on CPU.
```


### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)
