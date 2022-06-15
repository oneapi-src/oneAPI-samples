# Getting Started Samples for Intel® oneAPI Rendering Toolkit (Render Kit)

The Intel&reg; oneAPI Rendering Toolkit (Render Kit) is designed to accelerate photorealistic rendering workloads with rendering and ray-tracing libraries to create high-performance, high-fidelity visual experiences. You can get the most from Intel&reg; hardware by optimizing performance at any scale with these libraries. Creators, scientists, and engineers can push the boundaries of visualization by using the toolkit to develop studio animation and visual effects or to create scientific and industrial visualizations.

You can find more information at [Intel&reg; oneAPI Rendering Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/rendering-toolkit.html).

| Minimum Requirements              | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 <BR>CentOS 8 (or compatible) <BR>Windows* 10 <BR>macOS* 10.15+
| Hardware                          | Intel 64 Penryn or newer with SSE4.1 extensions, ARM64 with NEON extensions<br>(Optimized requirement: Intel 64 Skylake or newer with AVX512 extensions, ARM64 with NEON extensions)
| Compiler Toolchain                | Windows OS: MSVS 2019 with Windows SDK and CMake*<BR>Other platforms: C++11 compiler, a C99 compiler (for example, gcc/c++/clang), and CMake*
| Libraries                         | Install Intel&reg; oneAPI Rendering Toolkit, including Intel&reg; OSPRay, Intel&reg; Embree, Intel&reg; Open Volume Kernel Library (Intel&reg; Open VKL), and Intel&reg; Open Image Denoise
| Image Display Tool                | A .ppm and .pfm filetype viewer (for example, [ImageMagick*](https://www.imagemagick.org)).
| Image Conversion Tool             | A converter for .ppm, .pfm, and endian conversions (for example, ImageMagick).


| Objective                         | Description
|:---                               |:---
| What you will learn               | How to build and run sample programs for the component libraries in Render Kit.
| Time to complete                  | 20 minutes

## Purpose

The getting started samples demonstrate an ordered source code introduction to the functionality of Render Kit libraries.
These samples supplement the Get Started Guides:
- [Get Started with the Intel&reg; oneAPI Rendering Toolkit for Windows](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-oneapi-render-windows/top.html)
- [Get Started with the Intel&reg; oneAPI Rendering Toolkit for Linux](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-oneapi-render-linux/top.html)
- [Get Started with the Intel&reg; oneAPI Rendering Toolkit for macOS](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-oneapi-render-macos/top.html)


## Build and Run Render Kit Samples

Try the getting started sample programs in the order shown in the table. The output of a sample may serve as input for another sample. Use the specific README.md of each for instructions.

| Order | Component      | Folder                                             | Description |
| -- | --------- | ------------------------------------------------ | - |
| 1 | Intel OSPRay | [01_ospray_gsg](01_ospray_gsg)                     | Get started with Intel&reg; OSPRay, an open source, scalable, and portable ray tracing engine for high-performance, high-fidelity visualization. |
| 2 | Intel Embree | [02_embree_gsg](02_embree_gsg)| Get started with Intel&reg; Embree, a collection of high-performance ray tracing kernels. |
| 3 | Intel Open Volume Kernel Library | [03_openvkl_gsg](03_openvkl_gsg)| Get started with Intel&reg; Open VKL, a collection of high-performance volume computation kernels. |
| 4 | Intel Open Image Denoise | [04_oidn_gsg](04_oidn_gsg) | Get started with Intel&reg; Open Image Denoise, an open source library of high-performance, high-quality, denoising filters for images rendered with ray tracing. |
| 5 | Intel ISPC | [05_ispc_gsg](05_ispc_gsg) | Get started with Intel&reg; Implicit SPMD Program Compiler (Intel&reg; ISPC), the C variant optimizing compiler used in conjunction with the Render Kit libraries. |

## License

Code samples are licensed under the Apache 2.0 license. See
[License.txt](LICENSE.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).