# `Computed Tomography Reconstruction` Sample

Computed Tomography shows how to use the Intel® oneAPI Math Kernel Library (oneMKL) DFT functionality to simulate computed tomography (CT) imaging.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04 <br> Windows* 10, 11
| Hardware            | Skylake with Gen9 or newer
| Software            | Intel® oneAPI Math Kernel Library (oneMKL)
| What you will learn | How to use oneMKL Discrete Fourier Transform (DFT) functionality
| Time to complete    | 15 minutes

For more information on oneMKL and complete documentation of all oneMKL routines, see https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-documentation.html.

## Purpose

Computed Tomography uses oneMKL discrete Fourier transform (DFT) routines to transform simulated raw CT data (as collected by a CT scanner) into a reconstructed image of the scanned object.

In computed tomography, the raw imaging data is a set of line integrals over the actual object, also known as its _Radon transform_. From this data, the original image must be recovered by approximately inverting the Radon transform. This sample uses Fourier reconstruction for inverting the Radon transform of a user-provided input image. Using batched 1D real DFT of Radon transform data points, samples of the input image's Fourier spectrum may be estimated on a polar grid. After interpolating the latter onto a Cartesian grid, an inverse 2D real DFT produces a fair reproduction of the original image.

This sample performs its computations on the default SYCL device. You can set the `ONEAPI_DEVICE_SELECTOR` environment variable to `*:cpu` or `*:gpu` to select the device to use.

## Key Implementation Details

To use oneMKL DFT routines, the sample creates double-precision real DFT descriptor objects and calls the `commit` member function with a `sycl::queue` object to define the device and context. The `compute_*` routines are then called to perform the actual computation with the appropriate descriptor object and input data.

## Using Visual Studio Code* (Optional)
You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel Software Developer Tools**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel Software Developer Tools**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see the
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).


## Building the Computed Tomography Reconstruction Sample
> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `$ bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\"Program Files (x86)"\Intel\oneAPI\setvars.bat`
> - For Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or MacOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### On a Linux* System
Run `make` to build and run the sample.

You can remove all generated files with `make clean`.

### On a Windows* System
Run `nmake` to build and run the sample. `nmake clean` removes temporary files.

> **Warning**: On Windows, static linking with oneMKL currently takes a very long time due to a known compiler issue. This will be addressed in an upcoming release.

## Running the Computed Tomography Reconstruction Sample

### Example of Output
If everything is working correctly, the example program will start with the 400x400 example image `input.bmp` then create simulated CT data from it (stored as `radon.bmp`). It will then reconstruct the original image in grayscale and store it as `restored.bmp`.

```
./computed_tomography
Reading original image from input.bmp
Generating Radon transform data from input.bmp
Saving Radon transform data in radon.bmp
Reconstructing image from the Radon projection data
        Step 1 - Batch of 400 real 1D in-place forward DFTs of length 400
        Step 2 - Interpolating spectrum from polar to cartesian grid
        Step 3 - In-place backward real 2D DFT of size 400x400
Saving restored image in restored.bmp
```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/current/overview.html).

## License

Code samples are licensed under the MIT license. See
[License.txt](License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](third-party-programs.txt).
