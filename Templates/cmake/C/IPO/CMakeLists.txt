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

project("ipo_c" LANGUAGES C)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_check_output LANGUAGES C)

if(ipo_supported)
    message(STATUS "Toolchain has C IPO support")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
else()
    message(WARNING "Toolchain WITHOUT IPO support: ${ipo_check_output}")
endif()

add_library(ipo_c_lib ipo_c_lib1.c ipo_c_lib.h)
add_executable(ipo_c main.c)
target_link_libraries(ipo_c PUBLIC ipo_c_lib)
