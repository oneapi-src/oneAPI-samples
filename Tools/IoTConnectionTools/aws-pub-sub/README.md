# `AWS Pub Sub` Sample

`AWS Pub Sub` is a sample that could be used for a quick test of Amazon cloud libraries.


| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 16.04, Linux* Ubuntu* 18.04
| Software                          | C++ 11 or higher, CMake 3.1+, Clang 3.9+ or GCC 4.4+, AWS IoT Device SDK C++ v2
| What you will learn               | Use the Message Broker for AWS IoT to send and receive messages through an MQTT connection


This version of the sample has been tested on Ubuntu Linux. This sample requires additional system configuration when using Ubuntu OS. Instructions on how to install the custom provided all dependency libraries for Linux can be [found here]().

## Purpose
`AWS Pub Sub` is a simple program that helps user to execute the example of the AWS code and toconfigure and run Amazon Cloud services.

## Key Implementation Details
This sample uses the Message Broker for AWS IoT to send and receive messages through an MQTT connection.

##License
This sample is licensed under Apache License v2.0

## Building the `AWS Pub Sub` 

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


## Running the Sample
### Application Parameters

endpoint: the endpoint of the mqtt server not including a port
cert: path to your client certificate in PEM format. If this is not set you must specify use_websocket
key: path to your key in PEM format. If this is not set you must specify use_websocket
topic: topic to publish, subscribe to.
client_id: client id to use (optional)
ca_file: Optional, if the mqtt server uses a certificate that's not already in your trust store, set this.
	It's the path to a CA file in PEM format
use_websocket: if specified, uses a websocket over https (optional)
signing_region: used for websocket signer it should only be specific if websockets are used. (required for websockets)
proxy_host: if you want to use a proxy with websockets, specify the host here (optional).
proxy_port: defaults to 8080 is proxy_host is set. Set this to any value you'd like (optional).

### Example of Output
TBD
