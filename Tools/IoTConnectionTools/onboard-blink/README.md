# Onboard Blink

## Introduction
This is a simple sample you could use for a quick IoT board test.

## What it is
A blink sample, it turns the on-board LED on for one second, then off for one second, repeatedly.

## Hardware requirements
A board with an on-board LED or an accessible GPIO pin (connected to a LED).

## Supported boards
This sample has been tested on
- [UP Squared\* AI Vision Kit](https://software.intel.com/en-us/iot/hardware/up-squared-ai-vision-dev-kit)
- [IEI\* Tank AIoT Developer Kit](https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core)

The sample might need minor modifications depending on the board, pin and shield you are using.

## Software requirements
This sample is supported on Linux systems only.

This version of the sample has been tested on Ubuntu Linux. It requires the [Eclipse* MRAA library](https://github.com/eclipse/mraa).

This sample requires additional system configuration when using Ubuntu OS with the UP series boards. Instructions on how to install the custom provided Linux kernel with the required drivers can be [found here](https://wiki.up-community.org/Ubuntu#Ubuntu_18.04_installation_and_configuration).

## Setup
Create a new project using the Eclipse* IDE and after installing the Intel® oneAPI IoT Toolkit. Make sure the on-board LED or the external LED is connected to the board.

## Note
Accessing device sensors, including LEDs, requires MRAA I/O operations. Mraa I/O operations require permissions to UNIX character devices and sysfs classes not commonly granted to normal users by default.
To learn how to use I/O devices from user space with the UP series boards refer to [this link](https://wiki.up-community.org/Ubuntu#Enable_the_HAT_functionality_from_userspace).

## Disclaimer
IMPORTANT NOTICE: This software is sample software. It is not designed or intended for use in any medical, life-saving or life-sustaining systems, transportation systems, nuclear systems, or for any other mission-critical application in which the failure of the system could lead to critical injury or death. The software may not be fully tested and may contain bugs or errors; it may not be intended or suitable for commercial release. No regulatory approvals for the software have been obtained, and therefore software may not be certified for use in certain countries or environments.
