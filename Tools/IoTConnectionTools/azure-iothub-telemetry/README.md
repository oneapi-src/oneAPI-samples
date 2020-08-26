# Azure IoTHub Telemetry

## Introduction
This is a simple sample you could use for a quick test of Azure cloud services.

## What it is
This project demonstrates how to send messages from a single device to Microsoft Azure IoT Hub via chosen protocol.

## Hardware requirements
The minimum requirements are for the device platform to support can be [found here](https://github.com/Azure/azure-iot-sdk-c).

## Software requirements
This sample is supported on Linux systems only.

This version of the sample has been tested on Ubuntu Linux. This sample requires additional system configuration when using Ubuntu OS. Instructions on how to install the custom provided all dependency libraries for Linux can be [found here](https://github.com/Azure/azure-iot-sdk-c/blob/master/doc/ubuntu_apt-get_sample_setup.md).

## Setup
Create and configure Azure IoTHub on [Microsoft Azure page](https://portal.azure.com/#home).
Detailed instructions are on [Microsoft website](https://docs.microsoft.com/en-us/azure/iot-hub/iot-hub-create-through-portal).

Paste the Device Connection String into the following line:
`static const char* connectionString = "[device connection string]"`

Choose one of the protocols to connect: MQTT over websockets, AMQP, AMQP over websockets or HTTP by uncommenting one of the following strings (MQTT protocol is chosen by default):
`//#define SAMPLE_MQTT_OVER_WEBSOCKETS`
`//#define SAMPLE_AMQP`
`//#define SAMPLE_AMQP_OVER_WEBSOCKETS`
`//#define SAMPLE_HTTP`

Build and run the sample.

## Disclaimer
IMPORTANT NOTICE: This software is sample software. It is not designed or intended for use in any medical, life-saving or life-sustaining systems, transportation systems, nuclear systems, or for any other mission-critical application in which the failure of the system could lead to critical injury or death. The software may not be fully tested and may contain bugs or errors; it may not be intended or suitable for commercial release. No regulatory approvals for the software have been obtained, and therefore software may not be certified for use in certain countries or environments.
