# Hello IoT World

## Introduction
This is a simple sample you could use for a quick compiler test.

## What it is
This project outputs a "Hello World" message and lets users know if the Intel® C++ Compiler Classic was used
to compile the code sample.

## Hardware requirements
Any Intel® CPU

## Software requirements
Intel® C++ Compiler Classic

## How to build and run
### Linux CLI
Use the `oneapi-cli` utility from a terminal to download and create the sample at a location of your choice.
Source the `setvars.sh` script distributed with oneAPI to configure the compiler. By default this can be found under
`/opt/intel/inteloneapi`.
Use the following commands to build and run the sample:
```
mkdir build
cd build
cmake ..
make
make run
```
### Windows CLI
Use the `oneapi-cli.exe` utility from a `Developer Command Prompt for VS` to download and create the sample at a location of your choice.

*Note:* On Windows systems you will need "MSBuild Tools", "Windows 10 SDK" and "C++ CMake tools for Windows" as part of your installed Visual Studio components.

Source the `setvars.bat` script distributed with oneAPI to configure the compiler. By default this can be found under
`"C:\Program Files (x86)\IntelOneAPI\inteloneapi"`.
Use the following commands to build and run the sample:
```
mkdir build
cd build
cmake -G "NMake Makefiles" ..
nmake
nmake run
```
### IDE
Use the Samples Plugin for Eclipse or Visual Studio to create and run the sample.

You may need to source the `setvars` script distributed with oneAPI before launching the IDE to use the Intel® C++ Compiler Classic or make it available as a toolchain in the IDE.

### Additional Links
Access the Getting Started Guides with the following links:
 * [Linux\*](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-oneapi-iot-toolkit)
 * [Windows\*](https://software.intel.com/en-us/get-started-with-intel-oneapi-windows-get-started-with-the-intel-oneapi-iot-toolkit)

## Disclaimer
IMPORTANT NOTICE: This software is sample software. It is not designed or intended for use in any medical, life-saving or life-sustaining systems, transportation systems, nuclear systems, or for any other mission-critical application in which the failure of the system could lead to critical injury or death. The software may not be fully tested and may contain bugs or errors; it may not be intended or suitable for commercial release. No regulatory approvals for the software have been obtained, and therefore software may not be certified for use in certain countries or environments.
