## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.0)

project(minimal_CPU CXX)

set(DEPS_ROOT /data/renderkit/dependencies)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../bin)

##find_package(embree 3.12.0 REQUIRED)
find_package(embree REQUIRED)

ADD_EXECUTABLE(${PROJECT_NAME} ../src/minimal_CPU.cpp)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} embree)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PRIVATE ${DEPS_ROOT}/install/include)
