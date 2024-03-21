## Copyright 2009-2024 Intel Corporation
## SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.0)

project(minimal_shadow CXX)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

set(ONEAPI_ROOT "")
if($ENV{ONEAPI_ROOT})
        set(ONEAPI_ROOT $ENV{ONEAPI_ROOT})
else()
        if(WIN32)
                set(ONEAPI_ROOT "C:/Program Files (x86)/Intel/oneAPI")
        else()
                set(ONEAPI_ROOT /opt/intel/oneapi)
        endif()
endif($ENV{ONEAPI_ROOT})


message(STATUS "ONEAPI_ROOT: ${ONEAPI_ROOT}")

find_package(embree REQUIRED HINTS ${ONEAPI_ROOT}/embree/latest)

add_executable(${PROJECT_NAME} ../src/minimal_shadow.cpp)

target_link_directories(${PROJECT_NAME} PRIVATE ${ONEAPI_ROOT}/compiler/latest/lib)
#As of embree 4.3 for CPU, the embree sycl backend needs libraries:
#  sycl svml irng imf intlc
#Using the compiler link library directory in the search path allows GNU linker (system compiler) to link in
target_link_libraries(${PROJECT_NAME} embree)
