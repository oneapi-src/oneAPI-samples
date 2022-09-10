# `Sepia Filter` Sample

The `Sepia Filter` sample is a program demonstrating how to convert a color image to a Sepia tone
image, which is a monochromatic image with a distinctive Brown Gray color. The
sample program works by offloading the compute intensive conversion of each pixel to
Sepia tone using SYCL*-compliant code for CPU and GPU.

For comprehensive information about oneAPI programming, see the [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide). (Use search or the table of contents to find relevant information quickly.)

| Property  | Description
|:---       |:---
| What you will learn  | How to write a custom device selector class, offload compute intensive parts of an application, and measure kernel execution time
| Time to complete     | 20 minutes

## Purpose

The `Sepia Filter` sample is a SYCL-compliant application that accepts a color image as an input
and converts it to a sepia tone image by applying the sepia filter coefficients
to every pixel of the image. The sample offloads the compute intensive part of the application, the processing of individual pixels, to an accelerator with the help of lambda and functor kernels. The sample also demonstrates how to use a custom device selector, which sets precedence for a GPU device over other available devices on the system. The device selected for offloading the kernel is displayed in the output and the time taken to execute each of the kernels. The application outputs a sepia tone image of the input image.

The sample demonstrates:
- Writing a custom device selector class.
- Offloading compute intensive parts of the application using both lamba and functor kernels.
- Measuring kernel execution time by enabling profiling.


## Prerequisites

| Optimized for  | Description
|:---            |:---
| OS             | Ubuntu* 18.04 <br>Windows* 10
| Hardware       | Skylake with GEN9 or newer
| Software       | Intel&reg; oneAPI DPC++/C++ Compiler

## Key Implementation Details

The basic SYCL implementation explained in the code includes device selector,
buffer, accessor, kernel, and command groups. This sample demonstrates a custom device selector implementation by overwriting the SYCL device selector class, offloading computation using both lambda and functor kernels, and using event objects to time command group execution, enabling profiling.

## Setting Environment Variables

Configure Command-Line Interface (CLI) development for tools in the Intel&reg; oneAPI Toolkits using environment variables.

Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. Doing so ensures that your compiler, libraries, and tools are ready for development.

### Linux*

Source the script from the installation location, which is typically in one of these folders:

For system wide installations: `. /opt/intel/oneapi/setvars.sh`

For private installations: `. ~/intel/oneapi/setvars.sh`

> **Note**: If you are using a non-POSIX shell, such as csh, use the following command:
> - `$ bash -c 'source <install-dir>/setvars.sh ; exec csh'`

If environment variables are set correctly, you will see a confirmation message.

> **Note**: You can use [Modulefiles scripts](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-modulefiles-with-linux.html) to set up your development environment. The modulefiles scripts work with all Linux shells.

> **Note**: If you wish to fine tune the list of components and the version of those components, use a [setvars config file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html) to set up your development environment.


### Windows*
Run the setvars.bat script from the root folder of your oneAPI installation, which is typically:

`"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"`

For Windows PowerShell* users, execute this command: 

`cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`

If environment variables are set correctly, you will see a confirmation message.

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the Diagnostics Utility for Intel&reg; oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors. See the [Diagnostics Utility for Intel&reg; oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).



## Build the `Sepia Filter` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
>
> Windows:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
>
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### Include Files
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

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
   $ cd sepia-filter
   $ mkdir build
   $ cd build
   $ cmake ../.
   $ make
   ```
If an error occurs, you can get more details by running `make` with the `VERBOSE=1` argument:
```
make VERBOSE=1
```

### On Windows* Using Visual Studio* Version 2017 or Newer
- Build the program using VS2017 or newer.
    - Right-click on the solution file, and open using VS2017 or a later version of the IDE.
    - Right-click on the project in **Solution Explorer** and select **Rebuild**.
    - From the top menu, select **Debug > Start without Debugging**.

- Build the program using MSBuild
     - Open **x64 Native Tools Command Prompt for VS2017** or whatever is appropriate for your IDE version. 
     - Enter the following command: `MSBuild sepia-filter.sln /t:Rebuild /p:Configuration="Release"`


## Run the Sample
1. Run the program.
   ```
   $ make run
   ```
### Application Parameters
The Sepia-filter application expects a **.png** image as an input parameter. The sample distribution includes some sample images in the **/input** folder. An image file, **silverfalls1.png**, is specified as the default input image to be converted in the `CMakeList.txt` file in the **/src** folder.

By default, three output images are written to the same folder as the application.

> **Note**: There is a known limitation due to an issue in the `Level0` driver. The sepia-filter fails with the default `Level0` backend. A workaround is in place to enable the OpenCL backend.

### Run the `Sepia Filter` Sample in Intel&reg; DevCloud
If running a sample in the Intel&reg; DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel&reg; oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

## Example Output
```
Loaded image with a width of 3264, a height of 2448 and 3 channels
Running on Intel(R) Gen9
Submitting lambda kernel...
Submitting functor kernel...
Waiting for execution to complete...
Execution completed
Lambda kernel time: 10.5153 milliseconds
Functor kernel time: 9.99602 milliseconds
Sepia tone successfully applied to image:[input/silverfalls1.png]
```
## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).