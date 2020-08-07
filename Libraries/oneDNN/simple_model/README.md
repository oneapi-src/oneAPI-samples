# oneDNN Simple Model Sample

This sample is implemented in C++ and DPC++ and runs on CPU or GPU. The sample
also includes [a Jupyter Notebook](simple_model.ipynb) that
demonstrates how to port a oneDNN sample from CPU-only version to CPU & GPU
in Intel oneAPI DevCloud environment.

| Optimized for                      | Description
| :---                               | :---
| OS                                 | Linux Ubuntu 18.04; Windows 10
| Hardware                           | Kaby Lake with GEN9 or newer
| Software                           | Intel oneAPI Deep Neural Network Library (oneDNN), Intel oneAPI DPC++ Compiler, Intel oneAPI Threading Building Blocks (oneTBB), GNU Compiler , Intel C++ Compiler
| What you will learn                | run a simple convolutional model on Intel CPU or Intel GPU
| Time to complete                   | 15 minutes

## License

This code sample is licensed under MIT license.

## What You Will Learn

* How to run a simple convolutional network on Intel CPU or Intel GPU.
* How to compile examples with Intel oneAPI DPC++ Compiler, Intel C++ Compiler,
and GNU C++ Compiler
* How to switch between OpenMP and TBB for CPU parallelization
* How tensors are implemented and submitted to primitives.
* How primitives are created.
* How primitives are sequentially submitted to the network, where the output
from primitives is passed as input to the next primitive. The latter specifies
a dependency between the primitive input and output data.
* Specific 'inference-only' configurations.
* Limiting the number of reorders performed that are detrimental to performance.

## Pre-requisites

### Using Intel C++ Compiler

Using Intel C++ Compiler also requires the following component which is part of the [Intel oneAPI HPC Toolkit (HPC Kit)](https://software.intel.com/en-us/oneapi/hpc-kit)
*  oneAPI Intel C++ Compiler

### Using TBB for CPU parallelization

Using Threading Building Blocks also requires the following component which is part of the [Intel oneAPI Base Toolkit (Base Kit)](https://software.intel.com/en-us/oneapi/oneapi-kit)
*  Intel oneAPI Threading Building Blocks (oneTBB)

### GPU and CPU

The sample below require the following components which are part of the [Intel oneAPI Base Toolkit (Base Kit)](https://software.intel.com/en-us/oneapi/oneapi-kit)
*  Intel oneAPI Deep Neural Network Library (oneDNN)
*  Intel oneAPI DPC++ Compiler
*  Intel oneAPI DPC++ Library (oneDPL)
*  Intel oneAPI Threading Building Blocks (oneTBB)

The sample also requires OpenCL driver. Please refer [System Requirements](https://software.intel.com/en-us/articles/intel-oneapi-base-toolkit-system-requirements) for OpenCL driver installation.

## Building the sample for CPU and GPU

### CPU

#### Using GNU C++ Compiler

When compiled with GNU C++ Compiler this sample runs on Intel CPU and uses
GNU OpenMP runtime for parallelism.

##### on a Linux* System

Start with a clean console environment.

```
source ${INTEL_ONEAPI_INSTALL_FOLDER}/setvars.sh --dnnl-configuration=cpu_gomp
```

Make sure that both the enviroments of compiler and oneDNN are properly set up
before you process following steps. If setvars.sh complains "not found" for
compiler or oneDNN, please check your installation first.

```
cd oneapi-toolkit/oneDNN/oneDNN_CNN_INFERENCE_FP32
mkdir cpu_gomp
cd cpu_gomp
cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
make cnn-inference-f32-cpp
```

> NOTE: The source file `cnn_inference_f32.cpp` will be copied from
> `${INTEL_ONEAPI_INSTALL_FOLDER}/oneDNN/latest/cpu_gomp` to `cpu_gomp/src` folder.
> You can rebuild the sample by typing `make` in `cpu_gomp` folder.

#### Using Intel C++ Compiler

When compiled with Intel C++ Compiler this sample runs on Intel CPU and
uses Intel OpenMP for CPU parallelism.

##### on a Linux* System

Start with a clean console environment.

```
source ${INTEL_ONEAPI_INSTALL_FOLDER}/setvars.sh --dnnl-configuration=cpu_iomp
```

Make sure that both the enviroments of compiler and oneDNN are properly set up
before you process following steps. If setvars.sh complains "not found" for
compiler or oneDNN, please check your installation first.

```
cd oneapi-toolkit/oneDNN/oneDNN_CNN_INFERENCE_FP32
mkdir cpu_iomp
cd cpu_iomp
cmake .. -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc
make cnn-inference-f32-cpp
```

> NOTE: The source file `cnn_inference_f32.cpp` will be copied from
> `${INTEL_ONEAPI_INSTALL_FOLDER}/oneDNN/latest/cpu_iomp` to `cpu_iomp/src` folder.
> You can rebuild the sample by typing `make` in `cpu_iomp` folder.

#### Using TBB

oneDNN supports both Intel OpenMP and TBB for CPU parallelization.
You can switch to TBB runtime using steps below.

##### on a Linux* System

Start with a clean console environment.

```
source ${INTEL_ONEAPI_INSTALL_FOLDER}/setvars.sh --dnnl-configuration=cpu_tbb
```

Make sure that both the enviroments of compiler and oneDNN are properly set up
before you process following steps. If setvars.sh complains "not found" for
compiler or oneDNN, please check your installation first.

```
cd oneapi-toolkit/oneDNN/oneDNN_CNN_INFERENCE_FP32
mkdir cpu_tbb
cd cpu_tbb
cmake ..
make cnn-inference-f32-cpp
```

> NOTE: The source file `cnn_inference_f32.cpp` will be copied from
> `${INTEL_ONEAPI_INSTALL_FOLDER}/oneDNN/latest/cpu_tbb` to `cpu_tbb/src` folder.
You can rebuild the sample by typing `make` in `cpu_tbb` folder.

#### On a Windows* System 

When compiled with Microsoft Visual C++ Compiler this sample runs on Intel CPU
and uses Microsoft OpenMP runtime for parallelism.


Start with Intel oneAPI command prompt for Microsoft Visual Studio.

```
C:\Program Files (x86)\intel\oneapi> oneDNN\latest\env\vars.bat --dnnl-configuration=cpu_vcomp
```

Make sure that both the enviroments of compiler and oneDNN are properly set up
before you process following steps.

```
cd oneapi-toolkit/oneDNN/oneDNN_CNN_INFERENCE_FP32
mkdir cpu_vcomp
cd cpu_vcomp
cmake -G "Visual Studio 16 2019" ..
cmake --build .
```

> NOTE: You can open the oneDNN_CNN.sln inside cpu_vcomp folder to edit source
> code with Microsoft Visual Studio integrated development environment.

## CPU and GPU

### Using DPC++ Compiler

By using DPC++ compiler, this sample supports CNN FP32 both on Intel CPU and GPU.

#### on a Linux* System

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
cd oneapi-toolkit/oneDNN/oneDNN_CNN_INFERENCE_FP32
mkdir dpcpp
cd dpcpp
cmake .. -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=dpcpp
make cnn-inference-f32-cpp
```

> NOTE: The source file `cnn_inference_f32.cpp` will be copied from 
> `${INTEL_ONEAPI_INSTALL_FOLDER}/oneDNN/latest/dpcpp` to `dpcpp/src` folder.
You can rebuild the sample by typing `make` in `dpcpp` folder.

## Running the sample

### on a Linux* System

Run the program  on CPU:

```
./out/cnn-inference-f32-cpp
```

Run the program  on GPU:
```
./out/cnn-inference-f32-cpp gpu
```

> NOTE: Zero Level runtime is enabled by default. Please make sure proper
> installation of zero level driver
> including level-zero-devel package following installation guide.
> If you still encounter runtime issue such as "could not create a primitive",
> Please apply workaround to set SYCL_BE=PI_OPENCL before running
> a DPC++ program. For applying the workaround in this sample, users can add
> `export SYCL_BE=PI_OPENCL` in CMakeLists.txt. After applying the worklaround,
> sample use OpenCL runtime instead.

### On a Windows* System

Run the program  on CPU:

```
out\Debug\cnn-inference-f32-cpp.exe
```

### Example of Output

#### on a Linux* System

Enable oneDNN verbose log:

```
export DNNL_VERBOSE=1
```

Run the program on CPU or GPU following [How to Run Session](#how-to-run).

CPU Results:

```
dnnl_verbose,info,DNNL v1.90.1 (commit 9151ddc657e4c6775f17f3bcec46872e5fac47ee)
dnnl_verbose,info,Detected ISA is Intel AVX2
...
/oneDNN VERBOSE LOGS/
...
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:aBcd8b:f0 dst_f32::blocked:abcd:f0,,,1x256x6x6,0.032959
dnnl_verbose,exec,cpu,inner_product,gemm:jit,forward_inference,src_f32::blocked:abcd:f0 wei_f32::blocked:abcd:f0 bia_f32::blocked:a:f0 dst_f32::blocked:ab:f0,,,mb1ic256ih6iw6oc4096,5.4458
dnnl_verbose,exec,cpu,inner_product,gemm:jit,forward_inference,src_f32::blocked:ab:f0 wei_f32::blocked:ab:f0 bia_f32::blocked:a:f0 dst_f32::blocked:ab:f0,,,mb1ic4096oc4096,2.50317
dnnl_verbose,exec,cpu,inner_product,gemm:jit,forward_inference,src_f32::blocked:ab:f0 wei_f32::blocked:ab:f0 bia_f32::blocked:a:f0 dst_f32::blocked:ab:f0,,,mb1ic4096oc1000,0.634033
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:ab:f0 dst_f32::blocked:ab:f0,,,1x1000,0.0290527
Use time 33.22
```

GPU Results:

```
dnnl_verbose,info,DNNL v1.90.1 (commit 9151ddc657e4c6775f17f3bcec46872e5fac47ee)
dnnl_verbose,info,Detected ISA is Intel AVX2
...
/DNNL VERBOSE LOGS/
...
dnnl_verbose,exec,gpu,reorder,ocl:simple:any,undef,src_f32::blocked:aBcd16b:f0 dst_f32::blocked:abcd:f0,,,1x256x6x6
dnnl_verbose,exec,gpu,inner_product,ocl:gemm,forward_inference,src_f32::blocked:abcd:f0 wei_f32::blocked:abcd:f0 bia_f32::blocked:a:f0 dst_f32::blocked:ab:f0,,,mb1ic256ih6iw6oc4096
dnnl_verbose,exec,gpu,inner_product,ocl:gemm,forward_inference,src_f32::blocked:ab:f0 wei_f32::blocked:ab:f0 bia_f32::blocked:a:f0 dst_f32::blocked:ab:f0,,,mb1ic4096oc4096
dnnl_verbose,exec,gpu,inner_product,ocl:gemm,forward_inference,src_f32::blocked:ab:f0 wei_f32::blocked:ab:f0 bia_f32::blocked:a:f0 dst_f32::blocked:ab:f0,,,mb1ic4096oc1000
dnnl_verbose,exec,gpu,reorder,ocl:simple:any,undef,src_f32::blocked:ab:f0 dst_f32::blocked:ab:f0,,,1x1000
Use time 106.29
```

#### on a Windows* System

Enable oneDNN verbose log:
```
set DNNL_VERBOSE=1

```

Run the program on CPU or GPU following [How to Run Session](#how-to-run).

CPU Results:

```
dnnl_verbose,info,DNNL v1.90.1 (commit 9151ddc657e4c6775f17f3bcec46872e5fac47ee)
dnnl_verbose,info,Detected ISA is Intel AVX2
...
/DNNL VERBOSE LOGS/
...
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:aBcd8b:f0 dst_f32::blocked:abcd:f0,,,1x256x6x6,0.032959
dnnl_verbose,exec,cpu,inner_product,gemm:jit,forward_inference,src_f32::blocked:abcd:f0 wei_f32::blocked:abcd:f0 bia_f32::blocked:a:f0 dst_f32::blocked:ab:f0,,,mb1ic256ih6iw6oc4096,5.4458
dnnl_verbose,exec,cpu,inner_product,gemm:jit,forward_inference,src_f32::blocked:ab:f0 wei_f32::blocked:ab:f0 bia_f32::blocked:a:f0 dst_f32::blocked:ab:f0,,,mb1ic4096oc4096,2.50317
dnnl_verbose,exec,cpu,inner_product,gemm:jit,forward_inference,src_f32::blocked:ab:f0 wei_f32::blocked:ab:f0 bia_f32::blocked:a:f0 dst_f32::blocked:ab:f0,,,mb1ic4096oc1000,0.634033
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:ab:f0 dst_f32::blocked:ab:f0,,,1x1000,0.0290527
Use time 33.22
```

## Implementation Details

This sample uses example code from oneDNN distribution. You can find this code
in [oneDNN Github repository](https://github.com/oneapi-src/oneDNN/blob/dev-v2/examples/cnn_inference_f32.cpp).
