# Getting Started Sample for Intel oneAPI Rendering Toolkit: Intel Open Image Denoise

This sample program, oidnDenoise, denoises a raytraced image.

oidnDenoise takes a preprocessed image accumulatedFrameCpp for input. Recall, this image was originally generated from the Intel OSPRay getting started sample, ospTutorial.

oidnDenoise writes a denoised .pfm image file to disk.

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
- Open Image Denoise

Imaging Tools:
- An image **display program** for .ppm and .pfm filetypes . Ex: [ImageMagick](https://www.imagemagick.org/)
- An image **converter** for .ppm filetypes, .pfm filetypes, and endian conversions. Ex: [ImageMagick](https://www.imagemagick.org/)

## Build and Run

First, build and run the sample Intel OSPray getting started sample, ospTutorial, from the 01_ospray_gsg folder to generate input.

### Windows OS:


Run a new **x64 Native Tools Command Prompt for MSVS 2019**

```
call <path-to-oneapi-folder>\setvars.bat
cd <path-to-oneAPI-samples>\RenderingToolkit\oidn_gsg
mkdir build
cd build
cmake -G"Visual Studio 16 2019" -A x64 -DCMAKE_PREFIX_PATH="<path-to-oneapi-folder>" ..
cmake --build . --config Release
cd Release
```

Convert the accumulatedFrameCpp.ppm image to LSB data ordering and .pfm format. Example conversion with ImageMagick convert:

```
<path-to-ImageMagick>\magick convert ..\..\..\01_ospray_gsg\build\Release\accumulatedFrameCpp.ppm -endian LSB PFM:accumulatedFrameCpp.pfm
```

Denoise the image:

```
oidnDenoise.exe -hdr accumulatedFrameCpp.pfm -o denoised.pfm
```

Review the output for visual comparison to the input. Example view with ImageMagick display:

```
<path-to-ImageMagick>\imdisplay denoised.pfm
```



### Linux OS:

Start a new Terminal session
```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/oidn_gsg
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="<path-to-oneapi-folder>" ..
cmake --build .
```

Convert the accumulatedFrameCpp.ppm image to LSB data ordering and .pfm format. Example conversion with ImageMagick convert:
```
<path-to-ImageMagick>/magick convert ../../../01_ospray_gsg\build\Release\accumulatedFrameCpp.ppm -endian LSB PFM:accumulatedFrameCpp.pfm
```

Denoise the image:

```
./oidnDenoise -hdr accumulatedFrameCpp.pfm -o denoised.pfm
```

Review the output for visual comparison to the input. Example view with ImageMagick display:

```
<path-to-ImageMagick>/imdisplay denoised.pfm
```



### MacOS:

Start a new Terminal session

```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/oidn_gsg
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="<path-to-oneapi-folder>" ..
cmake --build .
```

Convert the accumulatedFrameCpp.ppm image to LSB data ordering and .pfm format. Example conversion with ImageMagick convert:
```
<path-to-ImageMagick>/magick convert ../../../01_ospray_gsg\build\Release\accumulatedFrameCpp.ppm -endian LSB PFM:accumulatedFrameCpp.pfm
```

Denoise the image:

```
oidnDenoise.exe -hdr accumulatedFrameCpp.pfm -o denoised.pfm
```

Review the output for visual comparison to the input. Example view with ImageMagick display:

```
<path-to-ImageMagick>/imdisplay denoised.pfm
```
