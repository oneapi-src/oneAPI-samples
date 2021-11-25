# Getting Started Sample for Intel oneAPI Rendering Toolkit: Intel Embree

Embree is a collection of high-performance ray tracing kernels. Improve performance of photo-realistic rendering applications by using performance optimized ray-tracing kernels.

| Minimum Requirements              | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04, CentOS 8 (or compatible); Windows 10; MacOS 10.15+
| Hardware                          | Intel 64 Penryn or newer with SSE4.1 extensions; ARM64 with NEON extensions
| Compiler Toolchain                | Windows* OS: MSVS 2019 with Windows* SDK and CMake*; Other platforms: C++11 compiler, a C99 compiler (ex: gcc/c++/clang), and CMake*
| Libraries                         | Install Intel oneAPI Rendering Toolkit including Embree

| Optimized Requirements            | Description
| :---                              | :---
| Hardware                          | Intel 64 Skylake or newer with AVX512 extentions, ARM64 with NEON extensions

| Objective                         | Description
|:---                               |:---
| What you will learn               | How to build and run a basic sample program using the Embree API from the Render Kit.
| Time to complete                  | 5 minutes

## Purpose

This sample program, `minimal`, performs two ray to triangle intersect tests with the Embree API. One test is successful, while the other test is a miss. Output is written to the console (stdout).

## Key Implementation Details

This source code is constructed to build with a C++ or a C compiler.

## License

This code sample is licensed under the Apache 2.0 license. See
[LICENSE.txt](LICENSE.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Build and Run

### Windows OS:

Run a new **x64 Native Tools Command Prompt for MSVS 2019**

```
call <path-to-oneapi-folder>\setvars.bat
cd <path-to-oneAPI-samples>\RenderingToolkit\02_embree_gsg
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd Release
minimal.exe
```

Review the terminal output (stdout)


### Linux OS:

Start a new Terminal session
```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/02_embree_gsg
mkdir build
cd build
cmake ..
cmake --build .
./minimal
```

Review the terminal output (stdout)


### MacOS:

Start a new Terminal session

```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/02_embree_gsg
mkdir build
cd build
cmake ..
cmake --build .
./minimal
```

Review the terminal output (stdout)
