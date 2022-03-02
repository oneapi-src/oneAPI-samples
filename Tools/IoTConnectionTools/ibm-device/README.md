# `IBM Device` Sample

`IBM Device` sample shows how to develop a device code using Watson IoT Platform iot-c device client library, connect and interact with Watson IoT Platform Service.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 16.04, Linux* Ubuntu* 18.04,
| Software                          | Paho MQTT C library, OpenSSL development package
| What you will learn               | Use protocol MQTT to send events from a device

## Purpose
This is a simple sample you could use for a test of the IBM device
connection. This project shows how to develop a device code using Watson IoT
Platform iot-c device client library, connect and interact with Watson IoT
Platform Service.

## Key Implementation Details
 This sample includes the function/code snippets to perform the following actions:
 - Initialize the client library
 - Configure device from configuration parameters specified in a configuration file
 - Set client logging
 - Enable error handling routines
 - Send device events to WIoTP service
 - Receive and process commands from WIoTP service

##License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Using Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

## Building the `IBM* Device` Sample

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
>For more information on environment variables, see Use the setvars Script for
[Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or
[Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### On a Linux* System

The detailed instructions on installing the custom kernel provided all
dependency libraries for Linux can be [found here](https://github.com/ibm-watson-iot/iot-c#build-instructions).

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
> **Note:** By default, the directory to install the 'iot-c' library is
> $ENV{HOME}. Otherwise, you should enter a valid path to this library
> as variable IOT_SDK_FOLDER in CMakeLists.txt file.

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

Configure the IoT device on [IBM Watson IoT Platform Page](https://ibm-watson-iot.github.io/iot-c/device/).

### Application Parameters

The samples uses the path to the configuration file as a parameter.
Download the configuration file with all the credentials according to [instructions](https://ibm-watson-iot.github.io/iot-c/device/).

### Example of Output

TBD

If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)

