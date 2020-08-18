cmake_minimum_required(VERSION 2.8.11)
project(azure-iothub-telemetry)

# Use the Intel C++ Compiler when available
find_program(ICPC NAMES icpc icl)
if(ICPC)
    set(CMAKE_CXX_COMPILER ${ICPC})
    message(STATUS "Switching to the Intel(R) C++ Compiler from: ${ICPC}")
else()
    message("-- Could not find the Intel(R) C++ Compiler on the system path. Did you setup the compiler environment?")
endif()


set(iothub_c_files
    iothub_ll_telemetry_sample.c
    ../certs/certs.c
)

add_definitions(-DUSE_HTTP)
add_definitions(-DUSE_AMQP)
add_definitions(-DUSE_MQTT)

add_definitions(-DSET_TRUSTED_CERT_IN_SAMPLES)
include_directories("../certs")



