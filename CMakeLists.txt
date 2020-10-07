# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE PATH "Build Type" FORCE)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(DPCPPSamples)
option(NODPL "Don't build samples requiring an installation of oneDPL and its dependencies")

enable_testing()

add_subdirectory(samples)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "Install Path" FORCE)
endif()
