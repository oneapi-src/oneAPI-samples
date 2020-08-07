# oneDNN Getting Started Sample

This sample is implemented in C++ and executes on CPU or GPU. The sample also
also includes [a Jupyer Notebook](getting_started.ipynb) that
demonstrates how to compile the code with various oneDNN configurations
in Intel oneAPI DevCloud environment.

| Optimized for                     | Description
| :---                              | :---
| OS                                | Linux Ubuntu 18.04; Windows 10
| Hardware                          | Kaby Lake with GEN9 or newer
| Software                          | Intel oneAPI Deep Neural Network Library (oneDNN), Intel oneAPI DPC++ Compiler, Intel oneAPI Threading Building Blocks (oneTBB)
| What you will learn               | basic oneDNN programming model for Intel CPU and GPU
| Time to complete                  | 15 minutes

## What You Will Learn

* How to create oneDNN memory objects.
* How to get data from application buffer into a oneDNN memory object.
* How tensor's logical dimensions and memory object formats relate.
* How to create oneDNN primitives.
* How to execute the primitives.

## Pre-requisites

The sample below require the following components, which are part of 
Intel oneAPI Base Toolkit (Base Kit):

* Intel oneAPI Deep Neural Network Library (oneDNN)
* Intel oneAPI DPC++ Compiler
* Intel oneAPI Threading Building Blocks (oneTBB)
* Intel Graphics Compute Runtime for oneAPI Level Zero and OpenCL Driver

Refer to [Intel oneAPI Toolkits Installation Guide](https://software.intel.com/content/www/us/en/develop/articles/installation-guide-for-intel-oneapi-toolkits.html)
for instructions on installing these components.

## Building the sample for CPU and GPU

### On a Linux* System

#### Using DPC++ Compiler

When compiled with Intel DPC++ Compiler this sample runs on Intel CPU
or Intel GPU and relies on Intel DPC++ Runtime for parallelism.



Start with a clean console environment.

```
source ${INTEL_ONEAPI_INSTALL_FOLDER}/setvars.sh
```

Specific oneDNN configuration may be selected with
`--dnnl-configuraition` option. Defailt configuration is `cpu_dpcpp_gpu_dpcpp`.

Make sure that both the enviroments of compiler and oneDNN are properly set up
before you process following steps. If setvars.sh complains "not found" for
compiler or oneDNN, please check your installation first.

```
cd oneapi-toolkit/oneDNN/oneDNN_Getting_Started
mkdir dpcpp
cd dpcpp
cmake .. -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=dpcpp
make getting-started-cpp
```

> NOTE: The source file `getting_started.cpp` will be copied from
>`${INTEL_ONEAPI_INSTALL_FOLDER}/oneDNN/latest/dpcpp` to `dpcpp/src folder`.
> You can rebuild the sample by typing `make` in `dpcpp` folder.

### On a Windows* System

When compiler with Microsoft C++ Compiler the sample runs on Intel CPU and uses
Microsoft OpenMP runtime for parallelism.

#### Visual Studio* Version 2015 or Newe

Start with Intel oneAPI command prompt for Microsoft Visual Studio.

```
C:\Program Files (x86)\intel\oneapi> oneDNN\latest\env\vars.bat --dnnl-configuration=cpu_vcomp
```

Make sure that both the enviroments of compiler and oneDNN are properly set up
before you process following steps.

```
cd oneapi-toolkit/oneDNN/oneDNN_Getting_Started
mkdir cpu_vcomp
cd cpu_vcomp
cmake -G "Visual Studio 16 2019" ..
cmake --build .
```

> NOTE: You can open the oneDNN_CNN.sln inside cpu_vcomp folder to edit source
> code with Microsoft Visual Studio integrated development environment.

## Running the Sample

### On a Linux* System

Run the program  on CPU

```
./out/getting-started-cpp cpu
```

Run the program  on GPU

```
./out/getting-started-cpp gpu
```

>  NOTE: Zero Level runtime is enabled by default. Please make sure proper
> installation of Level Zero driver including level-zero-devel package following
> installation guide. If you still encounter runtime issue such as "could not
> create a primitive", please apply workaround to set SYCL_BE=PI_OPENCL before
> running a DPC++ program. To apply the workaround in this sample add
> `export SYCL_BE=PI_OPENCL` in CMakeLists.txt. After applying the worklaround,
> the sample will use OpenCL runtime instead.

### On a Windows* System

Run the program  on CPU

```
out\Debug\getting-started-cpp.exe
```

### Example of Output

#### On a Linux* System

Enable oneDNN verbose log

```
export DNNL_VERBOSE=1
```

Run the program on CPU or GPU following [How to Run Session](#how-to-run)

CPU Results:

```
dnnl_verbose,info,DNNL v1.90.1 (commit 9151ddc657e4c6775f17f3bcec46872e5fac47ee)
dnnl_verbose,info,Detected ISA is Intel AVX2
dnnl_verbose,exec,cpu,eltwise,jit:avx2,forward_inference,data_f32::blocked:acdb:f0 diff_undef::undef::f0,,alg:eltwise_relu alpha:0 beta:0,1x3x13x13,704.982
Example passes
```

GPU Results:

```
dnnl_verbose,info,DNNL v1.90.1 (commit 9151ddc657e4c6775f17f3bcec46872e5fac47ee)
dnnl_verbose,info,Detected ISA is Intel AVX2
dnnl_verbose,exec,gpu,eltwise,ocl:ref:any,forward_inference,data_f32::blocked:acdb:f0 diff_undef::undef::f0,,alg:eltwise_relu alpha:0 beta:0,1x3x13x13
Example passes
```

#### On a Windows* System

Enable oneDNN verbose log

```
set DNNL_VERBOSE=1
```

Run the program on CPU or GPU following [How to Run Session](#how-to-run).

CPU Results:

```
dnnl_verbose,info,DNNL v1.90.1 (commit 9151ddc657e4c6775f17f3bcec46872e5fac47ee)
dnnl_verbose,info,Detected ISA is Intel AVX2
dnnl_verbose,exec,cpu,eltwise,jit:avx2,forward_inference,data_f32::blocked:acdb:f0 diff_undef::undef::f0,,alg:eltwise_relu alpha:0 beta:0,1x3x13x13,704.982
Example passes
```

## Implementation Details

This sample uses example code from oneDNN distribution. You can find this code
in [oneDNN Github repository](https://github.com/oneapi-src/oneDNN/blob/dev-v2/examples/getting_started.cpp).
