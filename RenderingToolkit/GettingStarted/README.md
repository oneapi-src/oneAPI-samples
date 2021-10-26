# Getting Started Samples for Intel® oneAPI Rendering Toolkit (RenderKit)

The Intel® oneAPI Rendering Toolkit is designed to accelerate photorealistic rendering workloads with rendering and ray-tracing libraries to create high-performance, high-fidelity visual experiences. With the libraries, get the most from Intel® hardware by optimizing performance at any scale. Creators, scientists, and engineers can push the boundaries of visualization by using the toolkit to develop studio animation and visual effects or to create scientific and industrial visualizations.

You can find more information at the [ Intel oneAPI Rendering Toolkit portal](https://software.intel.com/content/www/us/en/develop/tools/oneapi/rendering-toolkit.html).

| Minimum Requirements              | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04, CentOS 8 (or compatible); Windows 10; MacOS 10.15+
| Hardware                          | Intel 64 Penryn or newer with SSE4.1 extensions, ARM64 with NEON extensions
| Compiler Toolchain                | Windows* OS: MSVS 2019 with Windows* SDK and CMake*, Other platforms: C++11 compiler, a C99 compiler (ex: gcc/c++/clang), and CMake*
| Libraries                         | Install Intel oneAPI Rendering Toolkit including Intel OSPRay, Intel Embree, Intel Open Volume Kernel Library, Intel Open Image Denoise
| Image Display Tool                | A .ppm and .pfm filetype viewer. Ex: [ImageMagick](https://www.imagemagick.org)
| Image Conversion Tool             | A converter for .ppm, .pfm, and endian conversions. Ex: [ImageMagick](https://www.imagemagick.org)

| Optimized Requirements            | Description
| :---                              | :---
| Hardware                          | Intel 64 Skylake or newer with AVX512 extentions, ARM64 with NEON extensions

| Objective                         | Description
|:---                               |:---
| What you will learn               | How to build and run sample programs for the component libraries in Render Kit.
| Time to complete                  | 20 minutes

## Purpose

The getting started samples demonstrate an ordered source code introduction to the functionality of Render Kit libraries.
These samples supplement the Getting Started Guides:
- [Getting Started with the Intel oneAPI Rendering Toolkit for Windows* OS](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-oneapi-render-windows/top.html)
- [Getting Started with the Intel oneAPI Rendering Toolkit for Linux* OS](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-oneapi-render-linux/top.html)
- [Getting Started with the Intel oneAPI Rendering Toolkit for MacOS*](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-oneapi-render-macos/top.html)


## License

Code samples are licensed under the Apache 2.0 license. See
[License.txt](LICENSE.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Build and Run Render Kit Samples

Please try the getting started sample programs in order. Output of a sample may serve as input for another sample. Use the specific README.md of each for instructions.

| Order | Component      | Folder                                             | Description |
| -- | --------- | ------------------------------------------------ | - |
| 1 | Intel OSPRay | [01_ospray_gsg](01_ospray_gsg)                     | Get started with Intel OSPRay, an open source, scalable, and portable ray tracing engine for high-performance, high-fidelity visualization. |
| 2 | Intel Embree | [02_embree_gsg](02_embree_gsg)| Get started with Intel Embree, a collection of high-performance ray tracing kernels. |
| 3 | Intel Open Volume Kernel Library | [03_openvkl_gsg](03_openvkl_gsg)| Get started with Intel Open VKL, a collection of high-performance volume computation kernels. |
| 4 | Intel Open Image Denoise | [04_oidn_gsg](04_oidn_gsg) | Get started with Intel Open Image Denoise, an open source library of high-performance, high-quality, denoising filters for images rendered with ray tracing. |
