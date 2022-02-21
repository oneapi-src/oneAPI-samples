# `Hello IoT World` Sample

## Introduction

This source code is a simple sample you could use for a quick compiler test.

## What it is

This project outputs a "Hello World" message and lets users know if the Intel®
C++ Compiler Classic was used to compile the code sample.

## Hardware requirements

Any Intel® CPU

## Software requirements

Intel® C++ Compiler Classic

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Using Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to
this readme for instructions on how to build and run a sample.

## Build and Run

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
> For more information on environment variables, see Use the setvars Script for
> [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html),
> or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).



### Linux* CLI

Use the following commands to build and run the sample:

```
mkdir build
cd build
cmake ..
make
make run
```

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
``make VERBOSE=1``
For more comprehensive troubleshooting, use the Diagnostics Utility for
Intel® oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

### Eclipse* on Linux

Use the Intel Samples Plugin in Eclipse to create and run the sample.

You may need to source the `setvars.sh` script distributed with oneAPI before
launching Eclipse so that Eclipse can locate and use the Intel® C++ Compiler
Classic.

## Additional Links

Access the Getting Started Guides with the following links:

 * [Linux\*](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-oneapi-iot-linux/top.html)
 * [Windows\*](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-oneapi-iot-windows/top.html)

## Disclaimer

IMPORTANT NOTICE: This software is sample software. It is not designed or
intended for use in any medical, life-saving or life-sustaining systems,
transportation systems, nuclear systems, or for any other mission-critical
application in which the failure of the system could lead to critical injury
or death. The software may not be fully tested and may contain bugs or errors;
it may not be intended or suitable for commercial release. No regulatory
approvals for the software have been obtained, and therefore software may not
be certified for use in certain countries or environments.
