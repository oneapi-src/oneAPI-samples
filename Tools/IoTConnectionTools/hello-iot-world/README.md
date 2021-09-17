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

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Build and Run

### Linux CLI

Source the `setvars.sh` environment setup script distributed with oneAPI to
configure the oneAPI development environment. By default, this script can be
found in the `/opt/intel/oneapi` directory.

After sourcing `setvars.sh`, type `oneapi-cli` at the command line to download
and create the sample at a location of your choice.

Use the following commands to build and run the sample:

```
mkdir build
cd build
cmake ..
make
make run
```

### Eclipse on Linux

Use the Intel Samples Plugin in Eclipse to create and run the sample.

You may need to source the `setvars.sh` script distributed with oneAPI before
launching Eclipse so that Eclipse can locate and use the Intel® C++ Compiler
Classic.

## Additional Links

Access the Getting Started Guides with the following links:

 * [Linux\*](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-oneapi-iot-toolkit)
 * [Windows\*](https://software.intel.com/en-us/get-started-with-intel-oneapi-windows-get-started-with-the-intel-oneapi-iot-toolkit)

## Disclaimer

IMPORTANT NOTICE: This software is sample software. It is not designed or
intended for use in any medical, life-saving or life-sustaining systems,
transportation systems, nuclear systems, or for any other mission-critical
application in which the failure of the system could lead to critical injury
or death. The software may not be fully tested and may contain bugs or errors;
it may not be intended or suitable for commercial release. No regulatory
approvals for the software have been obtained, and therefore software may not
be certified for use in certain countries or environments.
