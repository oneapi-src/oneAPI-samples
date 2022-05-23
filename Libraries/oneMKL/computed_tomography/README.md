# `Computed Tomography Reconstruction` Sample

Computed Tomography shows how to use the oneMKL library's DFT functionality to simulate computed tomography (CT) imaging.

For more information on oneMKL, and complete documentation of all oneMKL routines, see https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04; Windows 10
| Hardware            | Skylake with Gen9 or newer
| Software            | Intel&reg; oneMKL
| What you will learn | How to use oneMKL's Discrete Fourier Transform (DFT) functionality
| Time to complete    | 15 minutes

## Purpose

Computed Tomography uses oneMKL's discrete Fourier transform (DFT) routines to transform simulated raw CT data (as collected by a CT scanner) into a reconstructed image of the scanned object.

In computed tomography, the raw imaging data is a set of line integrals over the actual object, also known as its _Radon transform_. From this data, the original image must be recovered by approximately inverting the Radon transform. This sample uses the filtered backprojection method for inverting the Radon transform, which involves a 1D DFT, followed by filtering, then an inverse 2D DFT to perform the final reconstruction.

This sample performs its computations on the default DPC++ device. You can set the `SYCL_DEVICE_TYPE` environment variable to `cpu` or `gpu` to select the device to use.

## Key Implementation Details

To use oneMKL DFT routines, the sample creates a descriptor object for the given precision and domain (real-to-complex or complex-to-complex), calls the `commit` method, and provides a `sycl::queue` object to define the device and context. The `compute_*` routines are then called to perform the actual computation with the appropriate descriptor object and input/output buffers.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

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

## Building the Computed Tomography Reconstruction Sample

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
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
Run `make` to build and run the sample.

You can remove all generated files with `make clean`.

### On a Windows* System
Run `nmake` to build and run the sample. `nmake clean` removes temporary files.

*Warning*: On Windows, static linking with oneMKL currently takes a very long time due to a known compiler issue. This will be addressed in an upcoming release.

## Running the Computed Tomography Reconstruction Sample

### Example of Output
If everything is working correctly, the example program will start with the 400x400 example image `input.bmp` then create simulated CT data from it (stored as `restored.bmp`). It will then reconstruct the original image in grayscale and store it as `restored.bmp`.

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

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)