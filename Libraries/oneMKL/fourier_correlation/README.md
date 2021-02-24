# 1D Fourier Correlation Sample
Fourier correlation has many applications, e.g.: measuring the similarity of two 1D signals, finding the best translation to overlay similar images, volumetric medical image segmentation, etc. This sample shows how to implement a 1D Fourier correlation using oneMKL kernel functions.

For more information on oneMKL, and complete documentation of all oneMKL routines, see https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04; Windows 10*
| Hardware            | Intel&reg; Skylake with Gen9 or newer
| Software            | Intel&reg; oneMKL
| What you will learn | How to implement a 1D Fourier correlation using oneMKL kernel functions
| Time to complete    | 15 minutes

## Purpose
This sample shows how to implement the Fourier correlation algorithm:

    corr = IDFT(DFT(signal1) * CONJG(DFT(signal1)))

Where ``DFT`` is the discrete Fourier transform, ``IDFT`` is the inverse DFT, and ``CONJG`` is the complex conjugate. 

The algorithm can be composed using oneMKL, which contains optimized forward and backward transforms and complex conjugate multiplication functions. Therefore, the entire computation can be performed on the accelerator device.

## Key Implementation Details
In many applications, only the final correlation result matters, so this is all that has to be transferred from the device back to the host. In this example, two artificial signals will be created on the device, transformed in-place, then correlated. The host will retrieve the final result and report the optimal translation and correlation score.

Two implementations of the Fourier correlation algorithm are provided: one that uses explicit buffering and one that uses Unified Shared Memory (USM). Both implementations perform the correlation on the selected device, but as an added bonus, the USM implementation performs the final maxloc reduction on the device rather than the host.

## License
Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building and Running the Fourier Correlation Sample

### Running Samples on the DevCloud
When running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/).

### On a Linux System
Run `make` to build and run the sample. Two programs are generated: one that uses explicit buffering and one that uses USM.

You can remove all generated files with `make clean`.

### On a Windows System
Run `nmake` to build and run the sample. 

Note: To removes temporary files, run `nmake clean`.

*Warning*: On Windows, static linking with oneMKL currently takes a very long time, due to a known compiler issue. This will be addressed in an upcoming oneMKL compiler release.

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
