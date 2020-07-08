# IBM Device

## Introduction
This is a simple sample you could use for a test of IBM device connection.

## What it is
This project shows how-to develop a device code using Watson IoT Platform iot-c device client library, connect and interact with Watson IoT Platform Service.

## Software requirements
This sample is supported on Linux systems only.

This version of the sample has been tested on Ubuntu Linux. This sample requires additional system configuration when using Ubuntu OS. Instructions on how to install the custom provided all dependency libraries for Linux can be [found here](https://github.com/ibm-watson-iot/iot-c#build-instructions).

## Setup
By default the directory to install the 'iot-c' library is $ENV{HOME}. Otherwise you should enter valid path to this library as variable IOT_SDK_FOLDER in CMakeLists.txt file.
Configure device on [IBM Watson IoT Platform Page](https://ibm-watson-iot.github.io/iot-c/device/).
Download the configuration file with all the credentials according to [instructions](https://ibm-watson-iot.github.io/iot-c/device/).

Build the sample executable. Enter the following line to run the sample:
`ibm-device deviceSample --config <path_to_downloaded_configuration_file>`

## Disclaimer
IMPORTANT NOTICE: This software is sample software. It is not designed or intended for use in any medical, life-saving or life-sustaining systems, transportation systems, nuclear systems, or for any other mission-critical application in which the failure of the system could lead to critical injury or death. The software may not be fully tested and may contain bugs or errors; it may not be intended or suitable for commercial release. No regulatory approvals for the software have been obtained, and therefore software may not be certified for use in certain countries or environments.
