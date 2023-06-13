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

# The SYCL community uses `add_sycl_to_target` to selectively compile a list
# of files with SYCL support.  Files not listed are not guaranted to be
# compiled with SYCL support, which may speed the compilation process.
#
# This CMake project demonstrates how to use `add_sycl_to_target` with the
# Intel oneAPI compilers in a way that should maximize compatibility with
# other SYCL implementations.
project(add-sycl-to-target-demo LANGUAGES CXX)

find_package(IntelSYCL REQUIRED)

# Compile specific sources for SYCL and build target for SYCL
add_executable(simple-add_sycl simple.cpp offload.cpp)
add_sycl_to_target(TARGET simple-add_sycl SOURCES offload.cpp)
