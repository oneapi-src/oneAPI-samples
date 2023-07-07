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

project("ipo_cxx" LANGUAGES CXX)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_check_output LANGUAGES CXX)

if(ipo_supported)
    message(STATUS "Toolchain has CXX IPO support")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
else()
    message(STATUS "Toolchain WITHOUT IPO support: ${ipo_check_output}")
endif()

add_library(ipo_cxx_lib ipo_cxx_lib1.cpp ipo_cxx_lib.h)
add_executable(ipo_cxx main.cpp)
target_link_libraries(ipo_cxx PUBLIC ipo_cxx_lib)
