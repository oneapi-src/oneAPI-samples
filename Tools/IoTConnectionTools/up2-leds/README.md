# `UP Squared* Built-in LEDs` Sample

## Introduction
This simple sample can be used to blink the built-in LEDs on the UP Squared board.

## What it is
This sample's primary purpose is to showcase the new LED class, and APIs added to the Eclipse* MRAA library.

## Hardware requirements
An [UP Squared](http://www.up-board.org/) board. No additional hardware is required.

## Supported boards
This sample is intended for the [UP Squared](http://www.up-board.org/) board.

This sample has been tested on
- [UP Squared\* AI Vision Kit](https://software.intel.com/en-us/iot/hardware/up-squared-ai-vision-dev-kit)

With minor modifications, it will run on any Linux board that exposes built-in LEDs using the gpio-leds
Linux kernel driver.

## Software requirements
This version of the sample has been tested on Ubuntu Linux but should be compatible with Ubilinux for the UP Squared.
It requires the [Eclipse* MRAA library](https://github.com/intel-iot-devkit/mraa) version 1.9.0 or newer.

This sample requires additional system configuration when using Ubuntu OS with the UP series boards. Instructions on installing the custom provided Linux kernel with the required drivers can be [found here](https://wiki.up-community.org/Ubuntu#Ubuntu_18.04_installation_and_configuration).

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Setup
Create a new project using this sample in Eclipse* IDE and install the Intel® oneAPI IoT Toolkit. Run it on the UP Squared board using a TCF connection.

## Note
Accessing device sensors, including LEDs, requires MRAA I/O operations. Mraa I/O operations require permissions to UNIX character devices and sysfs classes not commonly granted to normal users by default.
To learn how to use I/O devices from userspace with the UP series boards, refer to [this link](https://wiki.up-community.org/Ubuntu#Enable_the_HAT_functionality_from_userspace).

