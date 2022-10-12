# `AWS Pub Sub` Sample

`AWS Pub Sub` is a sample that could be used for a quick test of Amazon cloud libraries.


| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 16.04, Linux* Ubuntu* 18.04
| Software                          | C++ 11 or higher, CMake 3.1+, Clang 3.9+ or GCC 4.4+, AWS IoT Device SDK C++ v2
| What you will learn               | Use the Message Broker for AWS IoT to send and receive messages through an MQTT connection


This version of the sample has been tested on Ubuntu Linux. This sample
requires additional system configuration when using the Ubuntu OS. Instructions
on installing the custom provided all dependency libraries for Linux can be
[found here]().

## Purpose
`AWS Pub Sub` is a simple program that helps the user execute the AWS code
xample and configure and run Amazon Cloud services.

## Key Implementation Details
This sample uses the Message Broker for AWS IoT to send and receive messages
through an MQTT connection.

## License
This sample is licensed under Apache License v2.0

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

## Building the `AWS Pub Sub`

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

### On a Linux System

Perform the following steps:
1. Run in the terminal:
```
cd $HOME
mkdir sdk-cpp-workspace
cd sdk-cpp-workspace
git clone --recursive https://github.com/aws/aws-iot-device-sdk-cpp-v2.git
mkdir aws-iot-device-sdk-cpp-v2-build
cd aws-iot-device-sdk-cpp-v2-build
cmake -DCMAKE_INSTALL_PREFIX="<absolute path sdk-cpp-workspace dir>"  -DCMAKE_PREFIX_PATH="<absolute path sdk-cpp-workspace dir>" -DBUILD_DEPS=ON ../aws-iot-device-sdk-cpp-v2
cmake --build . --target install
```

2. To execute the sample that had been built run in the terminal:
```
basic-pub-sub --endpoint <endpoint> --cert <path to cert> --key <path to key> --topic --ca_file <optional: path to custom ca> --use_websocket --signing_region <region> --proxy_host <host> --proxy_port <port>
```

3. Clean the program using:

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

endpoint: the endpoint of the mqtt server not including a port
cert: path to your client certificate in PEM format. If this is not set, you must specify use_websocket
key: path to your key in PEM format. If this is not set, you must specify use_websocket
topic: topic to publish, subscribe to.
client_id: client id to use (optional)
ca_file: Optional, if the mqtt server uses a certificate that's not already in your trust store, set this.
	It's the path to a CA file in PEM format
use_websocket: if specified, uses a websocket over HTTPS (optional)
signing_region: used for websocket signer it should only be specific if websockets are used. (required for websockets)
proxy_host: if you want to use a proxy with websockets, specify the host here (optional).
proxy_port: defaults to 8080 is proxy_host is set. Set this to any value you'd like (optional).

### Example of Output
TBD
