# Getting Started Sample for Intel oneAPI Rendering Toolkit: Intel OSPRay

Intel OSPRay is an open source, scalable, and portable ray tracing engine for high-performance, high-fidelity visualization. Easily build applications that use ray tracing based rendering for both surface and volume-based visualizations. OSPRay builds on top of Embree, Open VKL, and Open Image Denoise.

| Minimum Requirements              | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04, CentOS 8 (or compatible); Windows 10; MacOS 10.15+
| Hardware                          | Intel 64 Penryn or newer with SSE4.1 extensions, ARM64 with NEON extensions
| Compiler Toolchain                | Windows* OS: MSVS 2019 installed with Windows* SDK and CMake*; Other platforms: C++11 compiler, a C99 compiler (ex: gcc/c++/clang), and CMake*
| Libraries                         | Install Intel oneAPI Rendering Toolkit including OSPRay, Embree, Open Volume Kernel Library, Intel Open Image Denoise
| Image Display Tool                | A .ppm filetype viewer. Ex: [ImageMagick](https://www.imagemagick.org)

| Optimized Requirements            | Description
| :---                              | :---
| Hardware                          | Intel 64 Skylake or newer with AVX512 extentions, ARM64 with NEON extensions

| Objective                         | Description
|:---                               |:---
| What you will learn               | How to build and run a basic rendering program using the OSPRay API from the Render Kit.
| Time to complete                  | 5 minutes


## Purpose

- This getting started sample program, `ospTutorialCpp`, renders two conjoined triangles with the [OSPRay API](https://www.ospray.org/documentation.html).
- Two renders of the triangles are written to .ppm image files on disk. The first image is rendered with one accumulation. The second image is rendered with ten accumulations.

## Key Implementation details

- The noise visible with only one accumulation is a common artifact of Monte Carlo based sampling. Notice the noise reduction (convergence) apparent in the image with ten accumulations.
- This sample uses the C++ API wrapper for the OSPRay API. The C++ API wrapper definitions are accessed via ospray_cpp.h. A pure C99 version of this tutorial is available on the [OSPRay github portal](https://github.com/ospray/ospray).
- This sample defines triangle vertex and color data using vector types found in the Render Kit rkcommon support library. These types can be swapped out for vector types found in the OpenGL* Math (GLM) library. An alternate, GLM, implementation of the sample is available for advanced users on the [OSPRay github portal](https://github.com/ospray/ospray).
- This sample renders single images. OSPRay is also used heavily in interactive rendering environments. Advanced users can see `ospExamples` on the [OSPRay github portal](https://github.com/ospray/ospray) and the [Intel OSPRay Studio](https://github.com/ospray/ospray_studio) showcase interactive application.

## License

This code sample is licensed under the Apache 2.0 license. See
[LICENSE.txt](LICENSE.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Build and Run

### Windows OS:

Run a new **x64 Native Tools Command Prompt for MSVS 2019**

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

Review the first output image with a .ppm image viewer. Example using ImageMagick display:
```
<path-to-ImageMagick>\imdisplay.exe firstFrameCpp.ppm
```

Review the accumulated output image with a .ppm image viewer. Example using ImageMagick display:
```
<path-to-ImageMagick>\imdisplay.exe accumulatedFrameCpp.ppm
```


### Linux OS:

Start a new Terminal session
```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/GettingStarted/01_ospray_gsg
mkdir build
cd build
cmake ..
cmake --build .
./ospTutorialCpp
```

Review the first output image with a .ppm image viewer. Example using ImageMagick display:
```
<path-to-ImageMagick>/display-im6 firstFrameCpp.ppm
```

Review the accumulated output image with a .ppm image viewer. Example using ImageMagick display:
```
<path-to-ImageMagick>/display-im6 accumulatedFrameCpp.ppm
```


### MacOS:

Start a new Terminal session

```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/GettingStarted/01_ospray_gsg
mkdir build
cd build
cmake ..
cmake --build .
./ospTutorialCpp
```

Review the first output image with a .ppm image viewer. Example using ImageMagick display:
```
<path-to-ImageMagick>/imdisplay firstFrameCpp.ppm
```

Review the accumulated output image with a .ppm image viewer. Example using ImageMagick display:
```
<path-to-ImageMagick>/imdisplay accumulatedFrameCpp.ppm
```
