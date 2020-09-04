# UP Squared* Built-in LEDs

## Introduction
This is a simple sample that can be used to blink the built-in LEDs on the UP Squared board.

## What it is
The main purpose of this sample is to showcase the new LED class and APIs added to the Eclipse* MRAA library.

## Hardware requirements
An [UP Squared](http://www.up-board.org/) board. No additional hardware required.

## Supported boards
This sample is intended for the [UP Squared](http://www.up-board.org/) board.

This sample has been tested on
- [UP Squared\* AI Vision Kit](https://software.intel.com/en-us/iot/hardware/up-squared-ai-vision-dev-kit)

With minor modifications, it will run on any Linux board that exposes built-in LEDs using the gpio-leds
Linux kernel driver.

## Software requirements
This version of the sample has been tested on Ubuntu Linux but should be compatible with Ubilinux for the UP Squared as well.
It requires the [Eclipse* MRAA library](https://github.com/intel-iot-devkit/mraa) version 1.9.0 or newer.

This sample requires additional system configuration when using Ubuntu OS with the UP series boards. Instructions on how to install the custom provided Linux kernel with the required drivers can be [found here](https://wiki.up-community.org/Ubuntu#Ubuntu_18.04_installation_and_configuration).

## Setup
Create a new project using this sample in Eclipse* IDE and after installing the Intel® oneAPI IoT Toolkit. Run it on the UP Squared board using a TCF connection.

## Note
Accessing device sensors, including LEDs, requires MRAA I/O operations. Mraa I/O operations require permissions to UNIX character devices and sysfs classes not commonly granted to normal users by default.
To learn how to use I/O devices from user space with the UP series boards refer to [this link](https://wiki.up-community.org/Ubuntu#Enable_the_HAT_functionality_from_userspace).

## Disclaimer
IMPORTANT NOTICE: This software is sample software. It is not designed or intended for use in any medical, life-saving or life-sustaining systems, transportation systems, nuclear systems, or for any other mission-critical application in which the failure of the system could lead to critical injury or death. The software may not be fully tested and may contain bugs or errors; it may not be intended or suitable for commercial release. No regulatory approvals for the software have been obtained, and therefore software may not be certified for use in certain countries or environments.
