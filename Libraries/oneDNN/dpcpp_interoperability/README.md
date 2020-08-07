# oneDNN DPC++ Interoperability Sample

This sample is implemented in DPC++ language and runs on CPU and GPU.

| Optimized for                      | Description
| :---                               | :---
| OS                                 | Linux Ubuntu 18.04;
| Hardware                           | Kaby Lake with GEN9 or newer
| Software                           | Intel oneAPI Deep Neural Network Library (oneDNN), Intel oneAPI DPC++ Compiler, Intel oneAPI Threading Building Blocks (oneTBB)
| What you will learn                | Using oneDNN in DPC++ application targeting Intel CPU or Intel GPU
| Time to complete                   | 15 minutes

## What You Will Learn

* How to create a GPU or CPU engine.
* How to create a memory descriptor/object.
* How to create a SYCL kernel for data initialization.
* How to access a SYCL buffer via SYCL interoperability interface.
* How to access a SYCL queue via SYCL interoperability interface.
* How to execute a SYCL kernel with related SYCL queue and SYCL buffer
* How to create operation descriptor/operation primitives descriptor/primitive.
* How to execute the primitive with the initialized memory.
* How to validate the result through a host accessor.

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

### on a Linux* System

#### Using DPC++ Compiler

When compiled with Intel oneAPI DPC++ Compiler this sample runs on Intel CPU
or Intel GPU.

Start with a clean console environment.

```
source ${INTEL_ONEAPI_INSTALL_FOLDER}/setvars.sh
```

Specific oneDNN configuration may be selected with
`--dnnl-configuraition` option. Defailt configuration is `cpu_dpcpp_gpu_dpcpp`.

Make sure that both the enviroments of compiler and oneDNN are properly set up
before you process following steps.
If setvars.sh complains "not found" for compiler or oneDNN, please check your
installation first.

```
cd oneapi-toolkit/oneDNN/oneDNN_SYCL_InterOp
mkdir dpcpp
cd dpcpp
cmake .. -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=dpcpp
make sycl-interop-cpp
```

> NOTE: The source file `sycl_interop.cpp` will be in `dpcpp/src` folder. 
> You can rebuild the sample by typing `make` in `dpcpp` folder.

## Running the Sample

### on a Linux* System
Run the program  on CPU:

```
./out/sycl-interop-cpp cpu
```

Run the program  on GPU

```
./out/sycl-interop-cpp gpu
```

>  NOTE: Zero Level runtime is enabled by default. Please make sure proper
> installation of Level Zero driver including level-zero-devel package following
> installation guide. If you still encounter runtime issue such as "could not
> create a primitive", please apply workaround to set SYCL_BE=PI_OPENCL before
> running a DPC++ program. To apply the workaround in this sample add
> `export SYCL_BE=PI_OPENCL` in CMakeLists.txt. After applying the worklaround,
> the sample will use OpenCL runtime instead.

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
dnnl_verbose,exec,cpu,eltwise,jit:avx2,forward_training,data_f32::blocked:abcd:f0 diff_undef::undef::f0,,alg:eltwise_relu alpha:0 beta:0,2x3x4x5,700.608
Example passes
```

GPU Results:

```
dnnl_verbose,info,DNNL v1.90.1 (commit 9151ddc657e4c6775f17f3bcec46872e5fac47ee)
dnnl_verbose,info,Detected ISA is Intel AVX2
dnnl_verbose,exec,gpu,eltwise,ocl:ref:any,forward_training,data_f32::blocked:abcd:f0 diff_undef::undef::f0,,alg:eltwise_relu alpha:0 beta:0,2x3x4x5
Example passes
```

## Implementation Details

This sample uses example code from oneDNN distribution. You can find this code
in [oneDNN Github repository](https://github.com/oneapi-src/oneDNN/blob/dev-v2/examples/sycl_interop.cpp).
