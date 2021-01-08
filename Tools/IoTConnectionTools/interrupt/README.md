# `ISR` Sample

## Introduction
You could use this simple sample for a quick test of an Interrupt Service Routine (ISR).

## What it is
Demonstrate how to react to an external event with an ISR (Interrupt Service Routine), which will run independently of the main program flow using the Eclipse* MRAA library.

## Hardware requirements
Use a platform with GPIO interrupt capabilities.
Any digital input or sensor that can generate a voltage transition from ground to Vcc or vice versa can be used with this example code.

## Supported boards
This sample has been tested on
- [UP Squared\* AI Vision Kit](https://software.intel.com/en-us/iot/hardware/up-squared-ai-vision-dev-kit)
- [IEI\* Tank AIoT Developer Kit](https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core)

The sample might need minor modifications depending on the board, pin and shield you are using.
*Note:* This sample does not work for the GPIO pins on the GrovePi+* shield.

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
Connect the input device to a GPIO pin on your IoT board (pin 13 is used by default on most boards).

## Note
Accessing device sensors, including LEDs, requires MRAA I/O operations. Mraa I/O operations require permissions to UNIX character devices and sysfs classes not commonly granted to normal users by default.
To learn how to use I/O devices from userspace with the UP series boards, refer to [this link](https://wiki.up-community.org/Ubuntu#Enable_the_HAT_functionality_from_userspace).

