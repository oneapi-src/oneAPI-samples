# Getting Started Sample for Intel&reg; Rendering Toolkit (Render Kit): Intel&reg; Open Volume Kernel Library (Intel&reg; Open VKL) on CPU

Intel&reg; Open Volume Kernel Library (Intel&reg; Open VKL) is a collection of
high-performance volume computation kernels. Improve performance of volume
rendering applications by using performance optimized volume traversal and
sampling functionality for a variety of data formats.

| Minimum Requirements              | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 22.04 <br>CentOS 8 (or compatible) <br> Windows* 10 or 11<br>macOS* 10.15+
| Hardware                          | Intel 64 Penryn or newer with SSE4.1 extensions, ARM64 with NEON extensions <br>(Optimized requirements: Intel 64 Skylake or newer with AVX512 extentions, ARM64 with NEON extensions) 
| Compiler Toolchain                | Windows OS: MSVS 2022 (or 2019) installed with Windows* SDK and CMake*; Other platforms: C++11 compiler, a C99 compiler (for example. gcc/c++/clang), and CMake*
| Libraries                         | <ul><li>Install Intel&reg; oneAPI Base Toolkit (Base Kit) for DPC++ Runtimes</li><li>macOS: Base Kit not required</li><li>Install Intel&reg; Rendering Toolkit (Render Kit), including Intel&reg; Embree and Intel&reg; Open VKL</li></ul>

| Objective                         | Description
|:---                               |:---
| What you will learn               | How to build and run a basic rendering program using the Intel&reg; Open VKL API from the Render Kit.
| Time to complete                  | 5 minutes

## Purpose

This sample program, `vklTutorialCPU`, shows sampling amongst a proceedurally
generated volume the different volumetric sampling capabilities with Intel&reg;
Open VKL. Output is written to the console (stdout).

## Key Implementation Details

`vklTutorialCPU` is written in C99 and is constructed to compile with a C++ or C99
compiler.

## Build and Run

### Windows

1. Run a new **x64 Native Tools Command Prompt for MSVS 2022**.

```
call <path-to-oneapi-folder>\setvars.bat
cd <path-to-oneAPI-samples>\RenderingToolkit\GettingStarted\03_openvkl_gsg
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd Release
vklTutorialCPU.exe
```

Note: MSVS 2019 should use an **x64 Native Tools Command Prompt for MSVS 2019**

2. Review the terminal output (stdout).


### Linux and macOS

1. Start a new Terminal session.
```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/GettingStarted/03_openvkl_gsg
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
./vklTutorialCPU
```

2. Review the terminal output (stdout).


## License

This code sample is licensed under the Apache 2.0 license. See
[LICENSE.txt](LICENSE.txt) for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
