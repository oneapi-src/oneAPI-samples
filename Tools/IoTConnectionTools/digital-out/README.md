# `Digital Out` Sample

## Introduction
This is a simple sample you could use for a quick test of digital output.

## What it is

This project demonstrates how to write a value to an output pin using the Eclipse* MRAA library.

## Hardware requirements
A board with an accessible GPIO output pin.
A digital output device or sensor such as a 'Relay' or 'LED' in Starter Kits.

## Supported boards
This sample has been tested on
- [UP Squared\* AI Vision Kit](https://software.intel.com/en-us/iot/hardware/up-squared-ai-vision-dev-kit)
- [IEI\* Tank AIoT Developer Kit](https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core)

The sample might need minor modifications depending on the board and shield you are using.

## Software requirements
This sample is supported on Linux systems only.

This version of the sample has been tested on Ubuntu Linux. It requires the [Eclipse* MRAA library](https://github.com/intel-iot-devkit/mraa).

This sample requires additional system configuration when using Ubuntu OS with the UP series boards. Instructions on installing the custom provided Linux kernel with the required drivers can be [found here](https://wiki.up-community.org/Ubuntu#Ubuntu_18.04_installation_and_configuration).

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Setup
Create a new project using this sample in Eclipse* IDE and install the Intel® oneAPI IoT Toolkit. 
Connect the output device to a digital output pin on your IoT board.

## Note
Accessing device sensors, including LEDs, requires MRAA I/O operations. Mraa I/O operations require permissions to UNIX character devices and sysfs classes not commonly granted to normal users by default.
To learn how to use I/O devices from userspace with the UP series boards, refer to [this link](https://wiki.up-community.org/Ubuntu#Enable_the_HAT_functionality_from_userspace).

