# `Sepia-filter` Sample

The sepia filter is a program that converts a color image to a Sepia tone
image, which is a monochromatic image with a distinctive Brown Gray color. The
program works by offloading the compute intensive conversion of each pixel to
Sepia tone and is implemented using DPC++ for CPU and GPU.

For comprehensive instructions see the [DPC++ Programming](https://software.intel.com/en-us/oneapi-programming-guide) and search based on relevant terms noted in the comments.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04, Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn               | The Sepia Filter sample demonstrates the following using the Intel&reg; oneAPI DPC++/C++ Compiler <ul><li>Writing a custom device selector class</li><li>Offloading compute intensive parts of the application using both lamba and functor kernels</li><li>Measuring kernel execution time by enabling profiling</li></ul>
| Time to complete                  | 20 minutes

## Purpose

The sepia filter is a DPC++ application that accepts a color image as an input
and converts it to a sepia tone image by applying the sepia filter coefficients
to every pixel of the image. The sample demonstrates offloading the compute
intensive part of the application, which is the processing of individual pixels
to an accelerator with lambda and functor kernels' help. The sample also
demonstrates the usage of a custom device selector, which sets precedence for a
GPU device over other available devices on the system. The device selected for
offloading the kernel is displayed in the output and the time taken to execute
each of the kernels. The application also outputs a sepia tone image of the
input image.

## Key implementation details

The basic DPC++ implementation explained in the code includes device selector,
buffer, accessor, kernel, and command groups. This sample also demonstrates a
custom device selector's implementation by overwriting the SYCL device selector
class, offloading computation using both lambda and functor kernels, and using
event objects to time command group execution, enabling profiling.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)



## Setting Environment Variables


For working at a Command-Line Interface (CLI), the tools in the oneAPI toolkits
are configured using environment variables. Set up your CLI environment by
sourcing the ``setvars`` script every time you open a new terminal window. This
will ensure that your compiler, libraries, and tools are ready for development.


### Linux
Source the script from the installation location, which is typically in one of
these folders:


For root or sudo installations:


  ``. /opt/intel/oneapi/setvars.sh``


For normal user installations:

  ``. ~/intel/oneapi/setvars.sh``

**Note:** If you are using a non-POSIX shell, such as csh, use the following command:

     ``$ bash -c 'source <install-dir>/setvars.sh ; exec csh'``

If environment variables are set correctly, you will see a confirmation
message.

If you receive an error message, troubleshoot the problem using the
Diagnostics Utility for Intel® oneAPI Toolkits, which provides system
checks to find missing dependencies and permissions errors.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).


**Note:** [Modulefiles scripts](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-modulefiles-with-linux.html)
    can also be used to set up your development environment.
    The modulefiles scripts work with all Linux shells.


**Note:** If you wish to fine
    tune the list of components and the version of those components, use
    a [setvars config file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html)
    to set up your development environment.

### Windows

Execute the  ``setvars.bat``  script from the root folder of your
oneAPI installation, which is typically:


  ``"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"``


For Windows PowerShell* users, execute this command:

  ``cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'``


If environment variables are set correctly, you will see a confirmation
message.

If you receive an error message, troubleshoot the problem using the
Diagnostics Utility for Intel® oneAPI Toolkits, which provides system
checks to find missing dependencies and permissions errors.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## Building the Program for CPU and GPU


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


### Include Files
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel&reg; oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)


### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

### On a Linux* System

Perform the following steps:


1.  Build the program using the following <code> cmake </code> commands.
```
    $ cd sepia-filter
    $ mkdir build
    $ cd build
    $ cmake ../.
    $ make
```

2.  Run the program <br>
```
    $ make run

```


If an error occurs, you can get more details by running `make` with the `VERBOSE=1` argument:
``make VERBOSE=1``
For more comprehensive troubleshooting, use the Diagnostics Utility for Intel® oneAPI Toolkits, which provides system checks to find missing dependencies and permissions errors.
[Learn more](https://software.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).


### On a Windows* System Using Visual Studio* Version 2017 or Newer
- Build the program using VS2017 or VS2019
    - Right-click on the solution file and open using either VS2017 or VS2019 IDE.
    - Right-click on the project in Solution Explorer and select Rebuild.
    - From the top menu, select Debug -> Start without Debugging.

- Build the program using MSBuild
     - Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019"
     - Run the following command: `MSBuild sepia-filter.sln /t:Rebuild /p:Configuration="Release"`


## Running the sample

### Application Parameters
The Sepia-filter application expects a png image as an input parameter. The application comes with some sample images in the input folder. One of these is specified as the default input image to be converted in the cmake file. The default output image is generated in the same folder as the application.

### Example Output
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
## Known Limitations

Due to a known issue in the Level0 driver, the sepia-filter fails with the default Level0 backend. A workaround is in place to enable the OpenCL backend.
