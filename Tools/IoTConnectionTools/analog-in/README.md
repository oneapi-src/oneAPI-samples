# Analog In

## Introduction
This is a simple sample you could use for a quick test of an analog input.

## What it is
This project demonstrates how to read an analog value from an input pin using the Eclipse* MRAA library.

## Hardware requirements
A board with an accessible GPIO input pin.
Some analog input device or sensor such as the Rotary Angle Sensor, Light Sensor, Sound Sensor, Temperature Sensor in Starter Kits.

## Supported boards
This sample has been tested on
- [UP Squared\* AI Vision Kit](https://software.intel.com/en-us/iot/hardware/up-squared-ai-vision-dev-kit)

The sample might need minor modifications depending on the board and shield you are using.

## Software requirements
This sample is supported on Linux systems only.

This version of the sample has been tested on Ubuntu Linux. It requires the [Eclipse* MRAA library](https://github.com/intel-iot-devkit/mraa).

This sample requires additional system configuration when using Ubuntu OS with the UP series boards. Instructions on how to install the custom provided Linux kernel with the required drivers can be [found here](https://wiki.up-community.org/Ubuntu#Ubuntu_18.04_installation_and_configuration).

In order to enable the ADC on Intel® Atom based UP boards, refer to the following [download](https://downloads.up-community.org/download/how-to-access-adc-for-up-squared-atom/).

## Setup
Create a new project using this sample in Eclipse* IDE and after installing the Intel® oneAPI IoT Toolkit.
Connect the input device to an analog input pin on your IoT board.

## Note
Accessing device sensors, including LEDs, requires MRAA I/O operations. Mraa I/O operations require permissions to UNIX character devices and sysfs classes not commonly granted to normal users by default.
To learn how to use I/O devices from user space with the UP series boards refer to [this link](https://wiki.up-community.org/Ubuntu#Enable_the_HAT_functionality_from_userspace).

## Disclaimer
IMPORTANT NOTICE: This software is sample software. It is not designed or intended for use in any medical, life-saving or life-sustaining systems, transportation systems, nuclear systems, or for any other mission-critical application in which the failure of the system could lead to critical injury or death. The software may not be fully tested and may contain bugs or errors; it may not be intended or suitable for commercial release. No regulatory approvals for the software have been obtained, and therefore software may not be certified for use in certain countries or environments.
