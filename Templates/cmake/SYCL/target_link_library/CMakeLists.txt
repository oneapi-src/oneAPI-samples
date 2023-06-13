#==============================================================
# Copyright © Intel Corporation
#
# SPDX-License-Identifier: MIT
#=============================================================

if (CMAKE_HOST_WIN32)
    # need CMake 3.25.0+ for IntelLLVM support of target link properties on Windows
    cmake_minimum_required(VERSION 3.25)
else()
    # CMake 3.20.5 is the minimum recommended for IntelLLVM on Linux
    cmake_minimum_required(VERSION 3.20.5)
endif()

# The Modern CMake way to use separate package is to set up CMake to use the
# package with `find_package`, then to add it to the project with
# `target_link_libraries`.  This project demonstrates how to build a SYCL
# target with Intel oneAPI compilers.
#
# While consistent with Modern CMake, this method does not work for other
# SYCL implementations.  Also, every source file in the specified target will
# be compiled for offload.  Compiling with offload enabled will not affect
# correctness but does lead to substantially longer compile times.
project(sycl-target-link-libraries LANGUAGES CXX)

find_package(IntelSYCL REQUIRED)

# Add SYCL properties to the whole target, the modern CMake way
add_executable(simple-tll simple.cpp)
target_link_libraries(simple-tll PRIVATE IntelSYCL::SYCL_CXX)
