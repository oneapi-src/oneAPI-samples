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


project(find_omp_cxx VERSION 1.0 LANGUAGES CXX)

find_package(OpenMP REQUIRED)

add_executable(hello-omp-cxx hello_omp.cpp)
target_link_libraries(hello-omp-cxx PUBLIC OpenMP::OpenMP_CXX)
