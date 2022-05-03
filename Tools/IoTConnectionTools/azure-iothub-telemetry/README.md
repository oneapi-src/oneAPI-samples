# `Azure Telemetry` Sample

`Azure Telemetry` sample demonstrates how to send messages from a single device to Microsoft Azure IoT Hub via a selected protocol.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 16.04, Linux* Ubuntu* 18.04,
| What you will learn               | Use one of the protocols to send events from a device

## Purpose
This simple code sample helps the user to test the advantages of the Azure cloud services.

## Key Implementation Details
This sample tests Azure Cloud IoT Hub. There are
five protocols to choose from; MQTT, AMQP, HTTP, MQTT over Websockets and AMQP
over Websockets. The sample requires an Azure account and created Azure IoT
Hub.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Using Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to
this readme for instructions on how to build and run a sample.


## Building the `Azure Telemetry` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

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

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
``make VERBOSE=1``
For more comprehensive troubleshooting, use the Diagnostics Utility for
Intel® oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

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
