# Getting Started Sample for Intel&reg; oneAPI Rendering Toolkit (Render Kit): Intel&reg; OSPRay

Intel&reg; OSPRay is an open source, scalable, and portable ray tracing engine
for high-performance, high-fidelity visualization. Easily build applications
that use ray tracing based rendering for both surface and volume-based
visualizations. OSPRay builds on top of Intel&reg; Embree, Intel&reg; Open
Volume Kernel Library (Intel&reg; Open VKL), and Intel&reg; Open Image Denoise.

| Minimum Requirements              | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 <br>CentOS 8 (or compatible) <br>Windows* 10 <br>macOS* 10.15+
| Hardware                          | Intel 64 Penryn or newer with SSE4.1 extensions, ARM64 with NEON extensions <br>(Optimized requirements: Intel 64 Skylake or newer with AVX512 extentions, ARM64 with NEON extensions)
| Compiler Toolchain                | Windows OS: MSVS 2019 installed with Windows SDK and CMake*; Other platforms: C++11 compiler, a C99 compiler (for example, gcc/c++/clang), and CMake*
| Libraries                         | Install Intel&reg; oneAPI Rendering Toolkit (Render Kit), including Intel&reg; OSPRay, Intel&reg; Embree, Intel&reg; Open VKL, and Intel&reg; Open Image Denoise
| Image Display Tool                | A .ppm filetype viewer (for example, [ImageMagick](https://www.imagemagick.org)).


| Objective                         | Description
|:---                               |:---
| What you will learn               | How to build and run a basic rendering program using the Intel&reg; OSPRay API from the Render Kit.
| Time to complete                  | 5 minutes


## Purpose

- This getting started sample program, `ospTutorialCpp`, renders two conjoined
  triangles with the [Intel&reg; OSPRay
  API](https://www.ospray.org/documentation.html).
- Two renders of the triangles are written to .ppm image files on disk. The
  first image is rendered with one accumulation. The second image is rendered
  with ten accumulations.

## Key Implementation details

- The noise visible with only one accumulation is a common artifact of Monte
  Carlo based sampling. Notice the noise reduction (convergence) apparent in the
  image with ten accumulations.
- This sample uses the C++ API wrapper for the Intel&reg; OSPRay API. The C++
  API wrapper definitions are accessed via ospray_cpp.h. A pure C99 version of
  this tutorial is available on the [OSPRay github
  portal](https://github.com/ospray/ospray).
- This sample defines triangle vertex and color data using vector types found in
  the Render Kit rkcommon support library. These types can be swapped out for
  vector types found in the OpenGL* Math (GLM) library. An alternate, GLM,
  implementation of the sample is available for advanced users on the [OSPRay
  github portal](https://github.com/ospray/ospray).
- This sample renders single images. OSPRay is also used heavily in interactive
  rendering environments. Advanced users can see `ospExamples` on the [OSPRay
  github portal](https://github.com/ospray/ospray) and the [Intel OSPRay
  Studio](https://github.com/ospray/ospray_studio) showcase interactive
  application.

## Build and Run

### Additional Notes

oneAPI Rendering Toolkit 2023.1 version's cmake file contains an errata. The errata will produce an error while building the example. Please apply the following workaround described in the following page. 2023.1.1 version will address the issue.

https://community.intel.com/t5/Intel-oneAPI-Rendering-Toolkit/2023-1-troubleshoot-errata-CMake-Error/m-p/1476040#M98
### Windows

1. Run a new **x64 Native Tools Command Prompt for MSVS 2019**.

```
call <path-to-oneapi-folder>\setvars.bat
cd <path-to-oneAPI-samples>\RenderingToolkit\GettingStarted\01_ospray_gsg
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd Release
ospTutorialCpp.exe
```

2. Review the first output image with a .ppm image viewer. Example using
   ImageMagick display:
```
<path-to-ImageMagick>\imdisplay.exe firstFrameCpp.ppm
```

3. Review the accumulated output image with a .ppm image viewer. Example using
   ImageMagick display:
```
<path-to-ImageMagick>\imdisplay.exe accumulatedFrameCpp.ppm
```


### Linux

1. Start a new Terminal session.
```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/GettingStarted/01_ospray_gsg
mkdir build
cd build
cmake ..
cmake --build .
./ospTutorialCpp
```

2. Review the first output image with a .ppm image viewer. Example using
   ImageMagick display:
```
<path-to-ImageMagick>/display-im6 firstFrameCpp.ppm
```

3. Review the accumulated output image with a .ppm image viewer. Example using
   ImageMagick display:
```
<path-to-ImageMagick>/display-im6 accumulatedFrameCpp.ppm
```
### macOS

1. Start a new Terminal session.

```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/GettingStarted/01_ospray_gsg
mkdir build
cd build
cmake ..
cmake --build .
./ospTutorialCpp
```

2. Review the first output image with a .ppm image viewer. Example using
   ImageMagick display:
```
<path-to-ImageMagick>/imdisplay firstFrameCpp.ppm
```

3. Review the accumulated output image with a .ppm image viewer. Example using
   ImageMagick display:
```
<path-to-ImageMagick>/imdisplay accumulatedFrameCpp.ppm
```

## License

This code sample is licensed under the Apache 2.0 license. See
[LICENSE.txt](LICENSE.txt) for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
