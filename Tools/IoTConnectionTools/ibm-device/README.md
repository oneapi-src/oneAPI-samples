# `IBM Device` Sample

`IBM Device` sample shows how-to develop a device code using Watson IoT Platform iot-c device client library, connect and interact with Watson IoT Platform Service.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 16.04, Linux* Ubuntu* 18.04,
| Software                          | Paho MQTT C library, OpenSSL development package
| What you will learn               | Use protocol MQTT to send events from a device

## Purpose
This is a simple sample you could use for a test of IBM device connection. This project shows how-to develop a device code using Watson IoT Platform iot-c device client library, connect and interact with Watson IoT Platform Service.

## Key Implementation Details
 This sample includes the function/code snippets to perform the following actions:
 - Initiliaze client library
 - Configure device from configuration parameters specified in a configuration file
 - Set client logging
 - Enable error handling routines
 - Send device events to WIoTP service
 - Receive and process commands from WIoTP service

##License
Code sample uses Eclipse Public License - v1.0

##Building the `IBM Device` Sample

### On a Linux* System

The detailed instructions on how to install the custom provided all dependency libraries for Linux can be [found here](https://github.com/ibm-watson-iot/iot-c#build-instructions).

Perform the following steps:
1. Run in the terminal:
    ```
    cd $ENV{HOME}
    git clone https://github.com/ibm-watson-iot/iot-c.git
    cd iot-c
    make
    sudo make -C paho.mqtt.c install
    sudo make install
    ```
> Note! By default the directory to install the 'iot-c' library is $ENV{HOME}. Otherwise you should enter valid path to this library as variable IOT_SDK_FOLDER in CMakeLists.txt file.

2. Run the following lines from the sample folder 'ibm-device':
    ```
    mkdir build
    cd build
    cmake ..
    make all
    ```
3. Run the program using:  
    ```
    ibm-device deviceSample --config <path_to_downloaded_configuration_file>
    ```
4. Clean the program using:
    ```
    make clean
    ```
## Running the Sample

Configure IoT device on [IBM Watson IoT Platform Page](https://ibm-watson-iot.github.io/iot-c/device/).

### Application Parameters

The samples uses path to configuration file as a parameter.
Download the configuration file with all the credentials according to [instructions](https://ibm-watson-iot.github.io/iot-c/device/).

### Example of Output

TBD
