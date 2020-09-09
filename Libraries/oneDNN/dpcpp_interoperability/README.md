# oneDNN DPC++ Interoperability Sample

This C++ API example demonstrates programming for Intel(R) Processor Graphics with SYCL extensions API in oneDNN. 
Users will know how to access SYCL buffer and queue via oneDNN SYCL interoperability interfaces,  
and this interface also helps users to execute a custom SYCL kernel with oneDNN library.

| Optimized for                      | Description
| :---                               | :---
| OS                                 | Linux Ubuntu 18.04;
| Hardware                           | Kaby Lake with GEN9 or newer
| Software                           | Intel oneAPI Deep Neural Network Library (oneDNN), Intel oneAPI DPC++/C++ Compiler, Intel oneAPI Threading Building Blocks (oneTBB)
| What you will learn                | Using oneDNN in DPC++ application targeting Intel CPU or Intel GPU
| Time to complete                   | 15 minutes

## Purpose

This sample demonstrates programming for Intel(R) Processor Graphics with SYCL extensions API in oneDNN.

With this sample you will learn:
* How to create a GPU or CPU engine.
* How to create a memory descriptor/object.
* How to create a SYCL kernel for data initialization.
* How to access a SYCL buffer via SYCL interoperability interface.
* How to access a SYCL queue via SYCL interoperability interface.
* How to execute a SYCL kernel with related SYCL queue and SYCL buffer
* How to create operation descriptor/operation primitives descriptor/primitive.
* How to execute the primitive with the initialized memory.
* How to validate the result through a host accessor.

The sample executes on system's CPU by default and can be executed on Intel GPU
using a command line parameter `gpu`.

## Key Implementation Details

This sample uses example file `${DNNLROOT}/examples/sycl_interop.cpp`
from oneDNN distribution. You can find this code in
[oneDNN Github repository](https://github.com/oneapi-src/oneDNN/blob/dev-v2/examples/sycl_interop.cpp).

Detailed code walkthrough is available in [oneDNN developer guide](https://oneapi-src.github.io/oneDNN/v2/sycl_interop_cpp.html)

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
./bin/sycl-interop-cpp
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
dnnl_verbose,exec,cpu,eltwise,jit:avx2,forward_training,data_f32::blocked:abcd:f0 diff_undef::undef::f0,,alg:eltwise_relu alpha:0 beta:0,2x3x4x5,0.36499
Example passed on CPU.
```

