# Computed Tomography Reconstruction Sample

Computed Tomography shows how to use the oneMKL library's DFT functionality to simulate computed tomography (CT) imaging.

For more information on oneMKL, and complete documentation of all oneMKL routines, see https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04; Windows 10
| Hardware            | Skylake with Gen9 or newer
| Software            | Intel&reg; oneMKL beta
| What you will learn | How to use oneMKL's Discrete Fourier Transform (DFT) functionality
| Time to complete    | 15 minutes

## Purpose

Computed Tomography uses oneMKL's discrete Fourier transform (DFT) routines to transform simulated raw CT data (as collected by a CT scanner) into a reconstructed image of the scanned object.

In computed tomography, the raw imaging data is a set of line integrals over the actual object, also known as its _Radon transform_. From this data, the original image must be recovered by approximately inverting the Radon transform. This sample uses the filtered backprojection method for inverting the Radon transform, which involves a 1D DFT, followed by filtering, then an inverse 2D DFT to perform the final reconstruction.

This sample performs its computations on the default DPC++ device. You can set the `SYCL_DEVICE_TYPE` environment variable to `cpu` or `gpu` to select the device to use.

## Key Implementation Details

To use oneMKL DFT routines, the sample creates a descriptor object for the given precision and domain (real-to-complex or complex-to-complex), calls the `commit` method, and provides a `sycl::queue` object to define the device and context. Then the `compute_*` routines are called to perform the actual computation with the appropriate descriptor object and input/output buffers.

## License

This code sample is licensed under the MIT license.

## Building the Computed Tomography Reconstruction Sample

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
Run `make` to build and run the sample.

You can remove all generated files with `make clean`.

### On a Windows* System
Run `nmake` to build and run the sample. `nmake clean` removes temporary files.

## Running the Computed Tomography Reconstruction Sample

### Example of Output
If everything is working correctly, the example program will start with the 400x400 example image `input.bmp`, create simulated CT data from it (stored as `restored.bmp`), then reconstruct the original image in grayscale and store it as `restored.bmp`.

```
./computed_tomography 400 400 input.bmp radon.bmp restored.bmp
Reading original image from input.bmp
Allocating radonImage for backprojection
Performing backprojection
Restoring original: step1 - fft_1d in-place
Allocating array for radial->cartesian interpolation
Restoring original: step2 - interpolation
Restoring original: step3 - ifft_2d in-place
Saving restored image to restored.bmp
```
