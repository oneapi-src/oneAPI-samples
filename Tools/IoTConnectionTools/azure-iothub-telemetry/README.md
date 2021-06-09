# `Azure Telemetry` Sample

`Azure Telemetry` sample demonstrates how to send messages from a single device to Microsoft Azure IoT Hub via a selected protocol.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 16.04, Linux* Ubuntu* 18.04,
| What you will learn               | Use one of the protocols to send events from a device

## Purpose
This simple code sample helps the user to test the advantages of the Azure cloud services.

## Key Implementation Details
This sample tests Azure Cloud IoT Hub. There are five protocols to choose from; MQTT, AMQP, HTTP, MQTT over Websockets and AMQP over Websockets.
The sample requires an Azure account and created Azure IoT Hub.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `Azure Telemetry` Sample

### On a Linux* System

Perform the following steps:

1. Create Azure IoT Hub using [the instruction](https://docs.microsoft.com/en-us/azure/iot-hub/iot-hub-create-through-portal) and copy the Primary Connection String.

2. Paste the Primary Connection String into the following line in the sample folder's file cpp/iothub_ll_telemetry_sample.c instead of the string in quotes:
    ```
    static const char* connectionString = "[device connection string]"
    ```

3. Add necessary PPAs and install all the prerequisite packages:
    ```
    sudo add-apt-repository -y ppa:mraa/mraa 
    sudo add-apt-repository -y ppa:aziotsdklinux/ppa-azureiot 
    sudo apt-get update 
    sudo apt-get install -y libmraa2 libmraa-dev libmraa-java python-mraa python3-mraa node-mraa mraa-tools pkg-config 
    sudo apt-get install -y azure-iot-sdk-c-dev
    ```

4. Run in the terminal:
    ```
    cd $ENV{HOME}
    git clone https://github.com/Azure/azure-iot-sdk-c.git
    cd azure-iot-sdk-c
    git submodule update --init
    mkdir cmake
    cd cmake
    cmake ..
    ```

5. Run the following lines from the sample folder 'azure-iot-telemetry':
    ```
    mkdir build
    cd build
    cmake ..
    make all
    ```
6. Run the program using:  
    ```
    make run
    ```
7. Clean the program using:
    ```
    make clean
    ```
## Running the Sample

### Application Parameters

There are no editable parameters for this sample.

### Example of Output
    ```
    Creating IoTHub Device handle
    The device client is connected to iothub

    Sending Message 1 to IoTHub
    Message:{"temperature": 24.716, "humidity":71.651, "scale":Celsius}
    confirmation callback received for message 1 with result IOTHUB_CLIENT_CONFIRMATION_OK

    Sending Message 2 to IoTHub
    Message:{"temperature": 31.408, "humidity":64.724, "scale":Celsius}
    confirmation callback received for message 2 with result IOTHUB_CLIENT_CONFIRMATION_OK

    Sending Message 3 to IoTHub
    Message:{"temperature": 26.158, "humidity":73.844, "scale":Celsius}
    confirmation callback received for message 3 with result IOTHUB_CLIENT_CONFIRMATION_OK

    Sending Message 4 to IoTHub
    Message:{"temperature": 21.599, "humidity":71.308, "scale":Celsius}
    confirmation callback received for message 4 with result IOTHUB_CLIENT_CONFIRMATION_OK
    ```
