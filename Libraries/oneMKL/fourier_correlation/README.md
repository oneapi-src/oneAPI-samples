# Fourier Correlation Sample
Fourier correlation has many applications, e.g.: measuring the similarity of two 1D signals, finding the best translation to overlay similar images, volumetric medical image segmentation, etc. This sample shows how to implement 1D and 2D Fourier correlation using SYCL, oneMKL, and oneDPL kernel functions.

For more information on oneMKL, and complete documentation of all oneMKL routines, see https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04; Windows 10*
| Hardware            | Intel&reg; Skylake with Gen9 or newer
| Software            | Intel&reg; oneMKL, Intel&reg; oneDPL
| What you will learn | How to implement the Fourier correlation algorithm using SYCL, oneMKL, and oneDPL functions
| Time to complete    | 15 minutes

## Purpose
This sample shows how to implement the Fourier correlation algorithm:

    corr = MAXLOC(IDFT(DFT(signal1) * CONJG(DFT(signal1))))

Where ``DFT`` is the discrete Fourier transform, ``IDFT`` is the inverse DFT, ``CONJG`` is the complex conjugate, and ``MAXLOC`` is the location of the maximum value.

The algorithm can be composed using SYCL, oneMKL, and/or oneDPL. SYCL provides the device offload and host-device memory transfer mechanisms. oneMKL provides optimized forward and backward transforms and complex conjugate multiplication functions. oneDPL provides the MAXLOC function. Therefore, the entire computation can be performed on the accelerator device.

The following articles provide more detailed explanations of the implementations:

 - [Efficiently Implementing Fourier Correlation Using oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-1/efficiently-implementing-fourier-correlation-using.html)
 - [Accelerating the 2D Fourier Correlation Algorithm with ArrayFire and oneAPI](https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-2d-fourier-correlation-algorithm.html)

## Key Implementation Details
In many applications, only the final correlation result matters, so this is all that has to be transferred from the device back to the host. In this example, two artificial signals will be created on the device, transformed in-place, then correlated. The host will retrieve the final result (i.e., the location of the maximum value) and report the optimal translation and correlation score.

Two implementations of the 1D Fourier correlation algorithm are provided: one that uses explicit buffering and one that uses Unified Shared Memory (USM). Both implementations perform the correlation on the selected device, but the buffered implementation uses the oneDPL max_element function to perform the final MAXLOC reduction while the USM implementation uses the SYCL reduction operator. A 2D Fourier correlation example is also included to show how 2D data layout is handled during the real-to-complex and complex-to-real transforms.

## License
Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Using Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

## Building and Running the Fourier Correlation Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat
>

>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2023-1/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2023-1/use-the-setvars-script-with-windows.html).

### Running Samples on the DevCloud
When running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/).


### On a Linux System
Run `make` to build and run the sample. Two programs are generated: one that uses explicit buffering and one that uses USM.

You can remove all generated files with `make clean`.

### On a Windows System
Run `nmake` to build and run the sample.

Note: To remove temporary files, run `nmake clean`.

### Example of Output
If everything is working correctly, the program will generate two artificial 1D signals, cross-correlate them, and report the relative shift that gives the maximum correlation score the output should be similar to this:
```
./fcorr_1d_buff 4096
Running on: Intel(R) Graphics Gen9 [0x3e96]
Shift the second signal 2048 elements relative to the first signal to get a maximum, normalized correlation score of 2.99976.
./fcorr_1d_usm 4096
Running on: Intel(R) Graphics Gen9 [0x3e96]
Shift the second signal 2048 elements relative to the first signal to get a maximum, normalized correlation score of 2.99975.
```
If the 2D example is working correctly, two small binary images are cross-correlated and their optimum relative shift and correlation score are reported:
```
./fcorr_2d_usm
Running on: Intel(R) UHD Graphics P630 [0x3e96]

First image:
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 1 1 0
0 0 0 0 0 1 1 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0

Second image:
0 0 0 0 0 0 0 0
0 1 1 0 0 0 0 0
0 1 1 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0

Normalized Fourier correlation result:
0 2.10734e-08 2.98023e-08 -2.10734e-08 -5.96046e-08 -2.10734e-08 2.98023e-08 2.10734e-08
7.45058e-09 1.58051e-08 2.98023e-08 -1.58051e-08 -6.70552e-08 -1.58051e-08 2.98023e-08 1.58051e-08
0 0 0 1 2 1 0 0
0 0 0 2 4 2 0 0
0 0 0 1 2 1 0 0
5.21541e-08 2.63418e-08 -2.98023e-08 -2.63418e-08 7.45058e-09 -2.63418e-08 -2.98023e-08 2.63418e-08
3.72529e-08 1.58051e-08 0 -1.58051e-08 -3.72529e-08 -1.58051e-08 0 1.58051e-08
1.49012e-08 -1.05367e-08 0 1.05367e-08 -1.49012e-08 1.05367e-08 0 -1.05367e-08

Shift the second image (x, y) = (4, 3) elements relative to the first image to get a maximum,
normalized correlation score of 4. Treat the images as circularly shifted versions of each other.
```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/2023-1/overview.html)
