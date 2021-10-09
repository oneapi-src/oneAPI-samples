# Getting Started Samples for Intel® oneAPI Rendering Toolkit (RenderKit)

TBD

You can find more information at the [ Intel oneAPI Rendering Toolkit portal](https://software.intel.com/content/www/us/en/develop/tools/oneapi/rendering-toolkit.html).

Users could learn how to run samples for different components in oneAPI Rendering Toolkit

## License

TBD

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Getting Started Samples

| Component      | Folder                                             | Description |
| --------- | ------------------------------------------------ | - |
| Intel OSPRay | [ospray_gsg](ospray_gsg)                     | Get started with Intel OSPRay |
| Intel Embree | [embree_gsg](embree_gsg)| Get started with Intel Embree |
| Intel Open Volume Kernel Library | [openvkl_gsg](openvkl_gsg)| Get started with Intel Open VKL |
| Intel Open Image Denoise | [oidn_gsg](oidn_gsg) | Get started with Intel Open Image Denoise |

## Build A Sample:

To build and run the samples you will need a compiler toolchain and imaging tools:

Compiler:
- MSVS2019 on Windows OS
- On other platforms a C++11 compiler and a C99 compiler. (Ex: gcc/g++/clang)

Imaging Tools:
- An image **display program** for .ppm and .pfm filetypes . Ex: [ImageMagick](https://www.imagemagick.org/)
- An image **converter** for .ppm and .pfm filetypes and endian conversions. Ex: [ImageMagick](https://www.imagemagick.org/)

### Windows OS:

Run a new **x64 Native Tools Command Prompt for VS 2019**

```
call <path-to-oneapi-folder>\setvars.bat
cd RenderingToolkit\<component-of-interest>_gsg
mkdir build
cd build
cmake -G”Visual Studio 16 2019” -A x64 -DCMAKE_PREFIX_PATH=”<path-to-oneapi-folder>” ..
cmake -–build . --config Release
cd Release
```


### Linux OS:

Start a new Terminal session
```
source <path-to-oneapi-folder>/setvars.sh
cd RenderingToolkit/<component-of-interest>_gsg
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=”<path-to-oneapi-folder>” ..
cmake -–build .
```



### MacOS:

Start a new Terminal session

```
source <path-to-oneapi-folder>/setvars.sh
cd RenderingToolkit/<component-of-interest>_gsg
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=”<path-to-oneapi-folder>” ..
cmake -–build .
```


## Run a built sample

### Running  ospray_gsg

Set environment variables if and only if they have not been already set in the same shell for the build step
- Windows OS: `call <path-to-oneapi-folder>\setvars.bat`
- Linux OS/MacOS: `source <path-to-oneapi-folder>/setvars.sh`
Run built '`ospTutorialCpp`' executable
Results are emmitted to disk in the local directory.
Use an image viewer like ImageMagick display (imdisplay) to review output .ppm files.
```
imdisplay firstFrameCpp.ppm
imdisplay accumulatedFrameCpp.ppm
```
Notice image convergence as OSPRay performs more accumulations to generate the second image.

### Running embree_gsg

Set environment variables if and only if they have not been already set in the same shell for the build step
- Windows OS: `call <path-to-oneapi-folder>\setvars.bat`
- Linux OS/MacOS: `source <path-to-oneapi-folder>/setvars.sh`

Run the built `minimal` executable
Embree results are emmitted to the command line interface.

### Running openvkl_gsg

Set environment variables if and only if they have not been already set in the same shell for the build step:
- Windows OS: `call <path-to-oneapi-folder>\setvars.bat`
- Linux OS/MacOS: `source <path-to-oneapi-folder>/setvars.sh`

Run the built `vklTutorial` executable from the command line
Results are emitted to the command line interface.

### Running oidn_gsg

Set environment variables if and only if they have not been already set in the same shell for the build step
- Windows OS: `call <path-to-oneapi-folder>\setvars.bat`
- Linux OS/MacOS: `source <path-to-oneapi-folder>/setvars.sh`

First, run ospray_gsg sample executable `ospTutorialCpp` from above.
Convert `ospTutorialCpp` output .ppm images to .pfm LSB format. Use a tool like [ImageMagick](https://www.imagemagick.org/) **convert**. Ex:

`magick convert accumulatedFrameCpp.ppm -endian LSB PFM:accumulatedFrameCpp.pfm`

Run oidn_gsg sample program `oidnDenoise` with .pfm as input:

`oidnDenoise --hdr <path-to-input-pfm-image>/accumulatedFrameCpp.pfm -o denoisedAccumulatedFrameCpp.pfm`

Review the output with an image viewer like [ImageMagick](https://www.imagemagick.org/) **display**:

`imdisplay denoisedAccumulatedFrameCpp.pfm`

Visually compare the denoised image to the original accumulatedFrameCpp.pfm:

`imdisplay <path-to-input-pfm-image>/accumulatedFrameCpp.pfm`







