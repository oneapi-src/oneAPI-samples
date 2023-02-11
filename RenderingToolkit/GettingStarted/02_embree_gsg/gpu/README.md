# Getting Started Sample for Intel&reg; oneAPI Rendering Toolkit (Render Kit): Intel&reg; Embree GPU

Intel Embree is a library of high-performance ray tracing kernels. Improve performance of photo-realistic rendering applications with Intel Embree.

| Minimum Requirements              | Description
|:---                               |:---
| OS                                | Ubuntu* 22.04, 20.04 <br> RHEL 8.5, 8.6 (or compatible) <br>Windows* 10 64-bit 20H2 or higher<br>Windows 11* 64-bit
| Hardware                          | Intel&reg; ARC GPU or higher, compatible with Intel Xe-HPG architecture
| Compiler Toolchain                | Windows: Intel&reg; oneAPI DPC++ Compiler and MSVS 2022 with Windows SDK and CMake*<br>Linux platforms: Intel&reg; oneAPI DPC++ Compiler, C++17 system compiler (for example g++), and CMake*
| Libraries                         | Install: <br>Intel&reg; oneAPI DPC++ Compiler and Runtime Library (Base Toolkit)<br>Intel&reg; oneAPI Rendering Toolkit (Render Kit), includes Embree

| Objective                         | Description
|:---                               |:---
| What you will learn               | How to build and run a basic sample program for Intel&reg; GPUs using the Embree API from the Render Kit.
| Time to complete                  | 5 minutes
| Configuration                     | See this [guide](https://dgpu-docs.intel.com/installation-guides/index.html#) for graphics driver install and configuration information

## Purpose

This sample program, `minimal_sycl`, performs two ray to triangle intersect tests
with the Intel&reg; Embree API. One test is a successful intersection, while the other test is
a miss.  Output is written to the console (stdout).

## Key Implementation Details

- This source code is written to build with Intel&reg; oneAPI Data Parallel C/C++ Compiler 2023.0.0 or higher.
- Embree computation is offloaded to the GPU device via the SYCL* implementation.
- The SYCL* implementation is provided with Intel DPC/C++ runtime libraries. It uses the system graphics driver.
- Embree releases with Beta GPU functionality report a Beta message.


## Build and Run

### Windows* OS

1. Run a new **x64 Native Tools Command Prompt for MSVS 2022**.

```
call <path-to-oneapi-folder>\setvars.bat
cd <path-to-oneAPI-samples>\RenderingToolkit\02_embree_gsg\gpu
mkdir build
cd build
cmake ..
cmake -G"Visual Studio 17 2022" -A x64 -T"Intel(R) oneAPI DPC++ Compiler 2023" ..
cmake --build . --config Release
cmake --install . --config Release
cd ..\bin
minimal_sycl.exe
```

2. Review the terminal output (stdout).
```
0.330000, 0.330000, -1.000000: Found intersection on geometry 0, primitive 0 at tfar=1.000000
1.000000, 1.000000, -1.000000: Did not find any intersection.

```

### Linux* OS

1. Start a new Terminal session.
```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/02_embree_gsg/gpu
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=icpx ..
cmake --build .
cmake --install .
cd ../bin
./minimal_sycl
```
2. Review the terminal output (stdout).
```
0.330000, 0.330000, -1.000000: Found intersection on geometry 0, primitive 0 at tfar=1.000000
1.000000, 1.000000, -1.000000: Did not find any intersection.

```

## Next Steps

Continue with the [Getting Started Guide](../../../GettingStarted/03_openvkl_gsg) or review the detailed [walkthrough tutorial programs](../../../Tutorial) using Embree to raytrace and pathtrace images.

## License

This code sample is licensed under the Apache 2.0 license. See
[LICENSE.txt](LICENSE.txt) for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
