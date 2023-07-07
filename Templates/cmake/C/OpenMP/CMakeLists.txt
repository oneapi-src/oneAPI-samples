#==============================================================
# Copyright © Intel Corporation
#
# SPDX-License-Identifier: MIT
#=============================================================

if (CMAKE_HOST_WIN32)
    # need CMake 3.25.0+ for IntelLLVM support of target link properties on Windows
    cmake_minimum_required(VERSION 3.25)
    if( NOT DEFINED CMAKE_C_COMPILER)
      set(CMAKE_C_COMPILER icx)
    endif()
    if( NOT DEFINED CMAKE_CXX_COMPILER)
      set(CMAKE_CXX_COMPILER icx)
    endif()
    if( NOT DEFINED CMAKE_Fortran_COMPILER)
      set(CMAKE_Fortran_COMPILER ifx)
    endif()
else()
    # CMake 3.20.5 is the minimum recommended for IntelLLVM on Linux
    cmake_minimum_required(VERSION 3.20.5)
    if( NOT DEFINED CMAKE_C_COMPILER)
      set(CMAKE_C_COMPILER icx)
    endif()
    if( NOT DEFINED CMAKE_CXX_COMPILER)
      set(CMAKE_CXX_COMPILER icpx)
    endif()
    if( NOT DEFINED CMAKE_Fortran_COMPILER)
      set(CMAKE_Fortran_COMPILER ifx)
    endif()
endif()

project(find_omp_c VERSION 1.0 LANGUAGES C)

find_package(OpenMP REQUIRED)

add_executable(hello-omp-c hello_omp.c)
target_link_libraries(hello-omp-c PUBLIC OpenMP::OpenMP_C)
