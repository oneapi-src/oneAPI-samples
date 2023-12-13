# Getting Started Sample for Intel&reg; Rendering Toolkit (Render Kit): Intel&reg; Open Volume Kernel Library (Intel&reg; Open VKL) for GPU

Intel&reg; Open Volume Kernel Library (Intel&reg; Open VKL) is a collection of
high-performance volume computation kernels. Improve performance of volume
rendering applications by using performance optimized volume traversal and
sampling functionality for a variety of data formats.

| Minimum Requirements              | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 22.04 <br>CentOS 8 (or compatible) <br> Windows* 10 or 11
| Hardware                          | Intel&reg; Arc&trade; GPU (DG2-128, DG2-512) or higher, compatible with Intel Xe-HPG or Intel Xe-HPC architectures
| Compiler Toolchain                | <ul><li>Windows* OS only: MSVS 2022 (or 2019) installed with Windows* SDK and CMake*</li><li> All Platforms: Intel&reg; oneAPI DPC++ Compiler from the Intel&reg; oneAPI Base Toolkit</li></ul>
| SYCL Compiler                     | oneAPI DPC++ 2024.0.0 compiler or higher
| Libraries                         | <ul><li>Install Intel&reg; Rendering Toolkit (Render Kit) including Intel&reg; Embree, Intel&reg; Open VKL</li><li> Install Intel&reg; oneAPI Base Toolkit (Base Kit) for Intel&reg; oneAPI DPC++ Compiler and Runtime Libraries</li></ul>
| GPU Configuration                 | **System BIOS**: [Quick Start](https://www.intel.com/content/www/us/en/support/articles/000091128/graphics.html) <br> **Windows\***: [Drivers for Intel&reg; Graphics products](https://www.intel.com/content/www/us/en/support/articles/000090440/graphics.html ) <br> **Linux\***: [Install Guide](https://dgpu-docs.intel.com/installation-guides/index.html#)
| Knowledge                         | First, build and run the [CPU](../cpu) get started program `vklTutorialCPU`

| Objective                         | Description
|:---                               |:---
| What you will learn               | How to build and run a basic rendering program using the Intel&reg; Open VKL API from the Render Kit targeting GPU
| Time to complete                  | 5 minutes

## Purpose

This sample program, `vklTutorialGPU`, shows sampling amongst a proceedurally
generated volume the different volumetric sampling capabilities with Intel&reg;
Open VKL. Output is written to the console (stdout).

## Key Implementation Details

`vklTutorialGPU` is written for C++17 with SYCL. The application is constructed to build with the Intel&reg; oneAPI DPC++ Compiler. On Windows* OS, it must also link to MSVS system libraries.

## Build and Run

### Windows

1. Run a new **x64 Native Tools Command Prompt for MSVS 2022**.

```
call <path-to-oneapi-folder>\setvars.bat
cd <path-to-oneAPI-samples>\RenderingToolkit\GettingStarted\03_openvkl_gsg\gpu
mkdir build
cd build
cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icx-cl ..
cmake --build .
.\vklTutorialGPU.exe
```

Note: MSVS 2019 users should use a **x64 Native Tools Command Prompt for MSVS 2019**

2. Review the terminal output (stdout).


### Linux

1. Start a new Terminal session.
```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/GettingStarted/03_openvkl_gsg/gpu
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx ..
cmake --build .
./vklTutorialGPU
```

2. Review the terminal output (stdout).


## License

This code sample is licensed under the Apache 2.0 license. See
[LICENSE.txt](LICENSE.txt) for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
