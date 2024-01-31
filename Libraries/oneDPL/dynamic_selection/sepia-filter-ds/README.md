# `Sepia Filter` Sample

The `Sepia Filter` sample is a program demonstrating how to convert a color image to a Sepia tone image, which is a monochromatic image with a distinctive Brown Gray color. The sample demonstrates how to use the Intel® oneAPI Base Toolkit (Base Kit) and Intel® oneAPI DPC++ Library (oneDPL) found in the Base Kit to easily apply Dynamic Device Selection policies that can help determine which device to run the application.

For comprehensive information about oneAPI programming, see the [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide). (Use search or the table of contents to find relevant information quickly.)

| Property  | Description
|:---       |:---
| What you will learn  | How to offload the computation to specific devices and use policies to different dynamic offload strategies.
| Time to complete     | 30 minutes

## Purpose

The `Sepia Filter` sample is a SYCL-compliant application that accepts a color image as an input and converts it to a sepia tone image by applying the sepia filter coefficients to every pixel of the image. The sample offloads the compute intensive part of the application, the processing of individual pixels, to an accelerator with the help of lambda and functor kernels.

This sample includes two versions:
1. `1_sepia_sycl.cpp`: basic SYCL implementation.
2. `2_sepia_policies.cpp`: Dynamic Device Selection iteration version that includes five policies:

  The Dynamic Device Selection API provides functions for choosing a resource using a selection policy. By default, the resources selected via these APIs in oneDPL are SYCL queues. There are several functions and selection policies provided as part of the API.

1. **Fixed CPU:** This is the simplest implementation. It can be helpful to start implementations using fixed CPU since any debug or troubleshooting will be considerably easier.
2. **Fixed GPU:** This an incremental step that simply designates the offload kernel to run on the GPU, isolating functionality to help triage any problems that may arise when targeting the GPU.
3. **Round Robin:** Assigns the function to the next available device as specified in the "universe". The capability is particularly beneficial in *multi-GPU systems*. Note that performance benefits may not be realized on single GPU platforms but will scale accordingly on multi-GPU systems.
4. **Dynamic Load** selects the device that has the most available capacity at that moment based on the number of unfinished submissions. This can be useful for offloading kernels of varying cost to devices of varying performance.
5. **Auto-tune** performs run-time profile sampling of the performance of the kernel on the available devices before selecting a final device to use. The choice is made based on runtime performance history, so this policy is only useful for kernels that have stable performance.

[Detailed Descriptions of the Policies](https://www.intel.com/content/www/us/en/docs/onedpl/developer-guide/current/policies.html) are available in the Intel® oneAPI DPC++ Library Developer Guide and Reference.

## Prerequisites

| Optimized for  | Description
|:---            |:---
| OS             | Ubuntu* 18.04 <br>Windows* 10
| Hardware       | Skylake with GEN9 or newer
| Software       | Intel&reg; oneAPI DPC++/C++ Compiler

## Key Implementation Details

The basic SYCL standards implemented in the code include the use of the following:
- Fixed (CPU and GPU) policies.
- Dynamic policies Round Robin, Load, and Auto-tune.
- Basic structure: header, namespace, define universe, setup policies, wrap kernel, and return event. **Note: a return event is required for all Dynamic Device Selection usage.**

The sample distribution includes some sample images in the **/input** folder. An image file, **silverfalls1.png**, is specified as the default input image to be converted in the `CMakeList.txt` file in the **/src** folder.

By default, three output images are written to the same folder as the application.

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the Diagnostics Utility for Intel&reg; oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors. See the [Diagnostics Utility for Intel&reg; oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## Build the `Sepia Filter` Sample

### Setting Environment Variables
When working with the Command Line Interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures your compiler, libraries, and tools are ready for development.

> **Note**: If you have not already done so, set up your CLI environment by sourcing the `setvars` script located in the root of your oneAPI installation.
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
> Microsoft Visual Studio:
> - Open a command prompt window and execute `setx SETVARS_CONFIG " "`. This only needs to be set once and will automatically execute the `setvars` script every time Visual Studio is launched.
>
>For more information on environment variables, see "Use the setvars Script" for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

You can use [Modulefiles scripts](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-modulefiles-with-linux.html) to set up your development environment. The modulefiles scripts work with all Linux shells.

If you wish to fine tune the list of components and the version of those components, use
a [setvars config file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html) to set up your development environment.

### Use Visual Studio Code* (Optional)
You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
- Download a sample using the extension **Code Sample Browser for Intel&reg; oneAPI Toolkits**.
- Configure the oneAPI environment with the extension **Environment Configurator for Intel&reg; oneAPI Toolkits**.
- Open a Terminal in VS Code (**Terminal>New Terminal**).
- Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*

Perform the following steps:

1. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make
   ```
If an error occurs, you can get more details by running `make` with the `VERBOSE=1` argument:
```
make VERBOSE=1
```

## Run the Sample
1. Run the program.

   Usage: ./1_sepia_sycl <num_images> <image_mix>
   ```
   $ ./1_sepia_sycl 1000 2
   ```
   This runs 1000 iterations of the base SYCL implementation applying the sepia filter to the larger of the supplied images using the default device for the platform.

   Image Mix:

   | Arg | Description
   |:--- |:---
   | 1   | Small images only
   | 2   | Large images only
   | 3   | 2 small : 2 large
   | 4   | 2 small : 1 large
   | 5   | 1 small : 2 large

2. Run the program using Dynamic Device Policies.

   The application requires arguments to determine the number of images (iterations), image size, and finally the Dynamic Device Policy to use. Note that for this sample application, a single image is provided in different sizes and is processed multiple times. Varying the image mix helps to mimic different workloads on your available hardware.

   Usage: ./2_sepia_policies <num_images> <image_mix> <policy>  

   For example:
   ```
   ./2_sepia_policies 1000 2 4
   ```
   The above runs a 1000 iterations of applying the sepia filter to the larger of the supplied images targetting the device with the most availalbe resources:

   Image Mix:

   | Arg | Description
   |:--- |:---
   | 1   | Small images only
   | 2   | Large images only
   | 3   | 2 small : 2 large
   | 4   | 2 small : 1 large
   | 5   | 1 small : 2 large

   | Arg | Dynamic Device Selection Policy
   |:--- |:---
   | 1   | Fixed Resource Policy (CPU)
   | 2   | Fixed Resource Policy (GPU)
   | 3   | Round Robin Policy
   | 4   | Dynamic Load Policy
   | 5   | Auto Tune Policy


### On Windows* Using Visual Studio* Version 2017 or Newer
- Build the program using VS2017 or newer.
    - Right-click on the solution file, and open using VS2017 or a later version of the IDE.
    - Right-click on the project in **Solution Explorer** and select **Rebuild**.
    - From the top menu, select **Debug > Start without Debugging**.

- Build the program using MSBuild
     - Open **x64 Native Tools Command Prompt for VS2017** or whatever is appropriate for your IDE version.
     - Enter the following command: `MSBuild sepia-filter.sln /t:Rebuild /p:Configuration="Release"`


## Example Output
Basic SYCL version:
```
$ ./1_sepia_sycl 1000 2
Processing 1000 images
Only large images
Will submit all offloads to default device: Intel(R) Iris(R) Xe Graphics [0x9a49]

Total time == 1772250 us
```
Dynamic Device Selection implementation:
```
 ./2_sepia_policies 1000 2 4
Processing 1000 images
Only large images
Using dynamic_load_policy to select least loaded device

Lambda running on Intel(R) Iris(R) Xe Graphics [0x9a49]
Lambda running on Intel(R) Iris(R) Xe Graphics [0x9a49]
...
Lambda running on Intel(R) Iris(R) Xe Graphics [0x9a49]
Lambda running on Intel(R) Iris(R) Xe Graphics [0x9a49]
Total time == 1871249 us
```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
