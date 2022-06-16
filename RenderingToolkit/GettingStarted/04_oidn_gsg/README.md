# Getting Started Sample for Intel&reg; oneAPI Rendering Toolkit (Render Kit): Intel&reg; Open Image Denoise


The Intel&reg; Open Image Denoise is an open source library of high-performance,
high-quality, denoising filters for images rendered with ray tracing. You can
significantly reduce rendering times in ray tracing based rendering applications
using the library.

| Minimum Requirements              | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 <br>CentOS 8 (or compatible) <br>Windows8 10 <br>macOS 10.15+
| Hardware                          | Intel 64 Penryn or newer with SSE4.1 extensions, ARM64 with NEON extensions <br>(Optimized requirements: Intel 64 Skylake or newer with AVX512 extentions, ARM64 with NEON extensions)
| Compiler Toolchain                | Windows* OS: MSVS 2019 installed with Windows* SDK and CMake*; Other platforms: C++11 compiler, a C99 compiler (ex: gcc/c++/clang), and CMake*
| Libraries                         | Install Intel&reg; oneAPI Rendering Toolkit (Render Kit),  including Intel&reg; OSPRay, Intel&reg; Embree, Intel&reg; Open Volume Kernel Library (Intel® Open VKL), and Intel&reg; Open Image Denoise
| Image Display Tool                | A .ppm filetype viewer (for example, [ImageMagick](https://www.imagemagick.org)).
| Image Conversion Tool             | A converter for .ppm, .pfm, and endian conversions (for example, ImageMagick).


| Objective                         | Description
|:---                               |:---
| What you will learn               | How to build and run a basic rendering program using the Intel&reg; Open Image Denoise API from the Render Kit.
| Time to complete                  | 5 minutes


## Purpose

This getting started sample program, `oidnDenoise`, denoises a raytraced image.
The output is written to disk as a .pfm image file.


## Key Implementation Details

- The program input is a noisy image. In this example, the `accumulatedFrameCpp`
  image is used for input. Recall, this image was originally generated from the
  Intel&reg; OSPRay getting started sample, `ospTutorial`. 
- The program writes a denoised .pfm image file to disk.
- Of course, oidnDenoise can denoise other, user-provided noisy input images.
  Along with the input image, the program can take in albedo and normal buffers
  corresponding to the same pixels of the input image. Inclusion of such
  auxialiary feature images can significantly improve denoising quality. See 
- The Intel&reg; OSPRay Studio showcase application demonstrates in-source
  denoising with the Intel&reg; OSPRay library. The noisy image buffer and
  auxiliary buffers are readily emitted from the Intel&reg; OSPRay API. All
  buffers are fed through the denoiser for a higher quality interactive
  experience.

## Build and Run

First, build and run the sample Intel&reg; OSPRay getting started sample,
`ospTutorial`, to generate input. Find it in the
[01_ospray_gsg](../01_ospray_gsg) folder of this samples repository.

### Windows

1. Run a new **x64 Native Tools Command Prompt for MSVS 2019**.

```
call <path-to-oneapi-folder>\setvars.bat
cd <path-to-oneAPI-samples>\RenderingToolkit\GettingStarted\04_oidn_gsg
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd Release
```

2. Convert the accumulatedFrameCpp.ppm image to LSB data ordering and .pfm
   format. Example conversion with ImageMagick convert:

```
<path-to-ImageMagick>\magick.exe convert <path-to-gsg>\01_ospray_gsg\build\Release\accumulatedFrameCpp.ppm -endian LSB PFM:accumulatedFrameCpp.pfm
```

3. Denoise the image.

```
oidnDenoise.exe -hdr accumulatedFrameCpp.pfm -o denoised.pfm
```

4. Review the output for visual comparison to the input. Example view with
   ImageMagick display:

```
<path-to-ImageMagick>\imdisplay.exe denoised.pfm
```

### Linux

1. Start a new Terminal session.
```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/GettingStarted/04_oidn_gsg
mkdir build
cd build
cmake ..
cmake --build .
```

2. Convert the `accumulatedFrameCpp.ppm` image to LSB data ordering and .pfm
   format. Example conversion with ImageMagick convert:
```
<path-to-ImageMagick>/convert-im6 <path-to-gsg>/01_ospray_gsg/build/accumulatedFrameCpp.ppm -endian LSB PFM:accumulatedFrameCpp.pfm
```

3. Denoise the image.

```
./oidnDenoise -hdr accumulatedFrameCpp.pfm -o denoised.pfm
```

4. Review the output for visual comparison to the input. Example view with
   ImageMagick display:

```
<path-to-ImageMagick>/display-im6 denoised.pfm
```



### macOS

1. Start a new Terminal session.

```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/GettingStarted/04_oidn_gsg
mkdir build
cd build
cmake ..
cmake --build .
```

2. Convert the `accumulatedFrameCpp.ppm` image to LSB data ordering and .pfm
   format. Example conversion with ImageMagick convert:
```
<path-to-ImageMagick>/magick convert <path-to-gsg>/01_ospray_gsg/build/accumulatedFrameCpp.ppm -endian LSB PFM:accumulatedFrameCpp.pfm
```

3. Denoise the image.

```
./oidnDenoise -hdr accumulatedFrameCpp.pfm -o denoised.pfm
```

4. Review the output for visual comparison to the input. Example view with
   ImageMagick display.

```
<path-to-ImageMagick>/imdisplay denoised.pfm
```
## License

This code sample is licensed under the Apache 2.0 license. See
[LICENSE.txt](LICENSE.txt) for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).