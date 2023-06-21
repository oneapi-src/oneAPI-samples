#==============================================================
# Copyright © Intel Corporation
#
# SPDX-License-Identifier: MIT
#=============================================================

if (CMAKE_HOST_WIN32)
    # need CMake 3.25.0+ for IntelLLVM support of target link properties on Windows
    cmake_minimum_required(VERSION 3.25)
    set(CMAKE_C_COMPILER icx)
    set(CMAKE_CXX_COMPILER icx)
    set(CMAKE_Fortran_COMPILER ifx)
else()
    # CMake 3.20.5 is the minimum recommended for IntelLLVM on Linux
    cmake_minimum_required(VERSION 3.20.5)
    set(CMAKE_C_COMPILER icx)
    set(CMAKE_CXX_COMPILER icpx)
    set(CMAKE_Fortran_COMPILER ifx)
endif()

project(find_omp_cxx VERSION 1.0 LANGUAGES CXX)

find_package(OpenMP REQUIRED)

add_executable(hello-omp-cxx hello_omp.cpp)
target_link_libraries(hello-omp-cxx PUBLIC OpenMP::OpenMP_CXX)
