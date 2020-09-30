cmake_minimum_required (VERSION 3.0)

project (ibm-device)

set(CMAKE_CXX_STANDARD "11")

set(IOT_SDK_FOLDER "$ENV{HOME}/iot-c")

include_directories(
  "${IOT_SDK_FOLDER}/paho.mqtt.c/src"
  "${IOT_SDK_FOLDER}/paho.mqtt.c/build/output"
  "${IOT_SDK_FOLDER}/build"
  "${IOT_SDK_FOLDER}/src/wiotp/sdk"
)

file(GLOB LIB_SRC ${IOT_SDK_FOLDER}/src/wiotp/sdk/*.c ${IOT_SDK_FOLDER}/src/wiotp/sdk/*.h)

add_library(iotp STATIC ${LIB_SRC})

set(device_sample_files cpp/deviceSample.c)

add_executable(ibm-device ${device_sample_files})

target_link_libraries(ibm-device iotp paho-mqtt3as pthread ssl crypto)

