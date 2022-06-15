# Getting Started Sample for Intel oneAPI Rendering Toolkit: Intel Open VKL

Intel Open VKL (Open Volume Kernel Library) is a collection of high-performance volume computation kernels. Improve performance of volume rendering applications by using performance optimized volume traversal and sampling functionality for a variety of data formats.

| Minimum Requirements              | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04, CentOS 8 (or compatible); Windows 10; MacOS 10.15+
| Hardware                          | Intel 64 Penryn or newer with SSE4.1 extensions, ARM64 with NEON extensions
| Compiler Toolchain                | Windows* OS: MSVS 2019 installed with Windows* SDK and CMake*; Other platforms: C++11 compiler, a C99 compiler (ex: gcc/c++/clang), and CMake*
| Libraries                         | Install Intel oneAPI Rendering Toolkit including Embree, Open Volume Kernel Library

| Optimized Requirements            | Description
| :---                              | :---
| Hardware                          | Intel 64 Skylake or newer with AVX512 extentions, ARM64 with NEON extensions

| Objective                         | Description
|:---                               |:---
| What you will learn               | How to build and run a basic rendering program using the Open VKL API from the Render Kit.
| Time to complete                  | 5 minutes

## Purpose

This sample program, `vklTutorial`, shows sampling amongst a proceedurally generated volume the different volumetric sampling capabilities with Intel Open VKL. Output is written to the console (stdout).

## Key Implementation Details

`vklTutorial` is written in C99 and is constructed to compile with a C++ or C99 compiler.

## License

This code sample is licensed under the Apache 2.0 license. See
[LICENSE.txt](LICENSE.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Build and Run

### Windows OS:

Run a new **x64 Native Tools Command Prompt for MSVS 2019**

```
call <path-to-oneapi-folder>\setvars.bat
cd <path-to-oneAPI-samples>\RenderingToolkit\GettingStarted\03_openvkl_gsg
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd Release
vklTutorial.exe
```

Review the terminal output (stdout)


### Linux OS:

Start a new Terminal session
```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/GettingStarted/03_openvkl_gsg
mkdir build
cd build
cmake ..
cmake --build .
./vklTutorial
```

Review the terminal output (stdout)


### MacOS:

Start a new Terminal session

```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/GettingStarted/03_openvkl_gsg
mkdir build
cd build
cmake ..
cmake --build .
./vklTutorial
```

Review the terminal output (stdout)
