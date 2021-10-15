# Getting Started Sample for Intel oneAPI Rendering Toolkit: Intel OSPRay

This sample ospTutorialCpp renders two triangles with the OSPRay API.

Two renders are written to .ppm image files to disk. The first image is rendered with one accumulation. The second image is rendered with 10 accumulations.


## License

TBD

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Requirements

To build and run the samples you will need a compiler toolchain and imaging tools:

Compiler:
- MSVS 2019 on Windows* OS
- On other platforms a C++11 compiler and a C99 compiler. (Ex: gcc/g++/clang)

oneAPI Libraries:
Install the Intel oneAPI Rendering Toolkit
- OSPRay
- Embree
- Open VKL

Imaging Tools:
- An image **display program** for .ppm and .pfm filetypes . Ex: [ImageMagick](https://www.imagemagick.org/)


## Build and Run

### Windows OS:

Run a new **x64 Native Tools Command Prompt for MSVS 2019**

```
call <path-to-oneapi-folder>\setvars.bat
cd <path-to-oneAPI-samples>\RenderingToolkit\ospray_gsg
mkdir build
cd build
cmake -G"Visual Studio 16 2019" -A x64 -DCMAKE_PREFIX_PATH="<path-to-oneapi-folder>" ..
cmake --build . --config Release
cd Release
ospTutorialCpp.exe
```

Review the first output image with a .ppm image viewer. Example using ImageMagick display:
```
<path-to-ImageMagick>\imdisplay firstFrameCpp.ppm
```

Review the accumulated output image with a .ppm image viewer. Example using ImageMagick display:
```
<path-to-ImageMagick>\imdisplay accumulatedFrameCpp.ppm
```


### Linux OS:

Start a new Terminal session
```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/ospray_gsg
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="<path-to-oneapi-folder>" ..
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


### MacOS:

Start a new Terminal session

```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/ospray_gsg
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="<path-to-oneapi-folder>" ..
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






