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

project(find_omp_fortran VERSION 1.0 LANGUAGES Fortran)

find_package(OpenMP REQUIRED)

add_library(pnm STATIC pnm.f90)

# The CPU version


# OpenMP target flags vary depending on platform
if (CMAKE_Fortran_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    # Flags to enable parallelization of DO CONCURRENT
    # In CMake 3.25.0+ adding OpenMP will make adding the flag unnecessary
    set(DO_CONCURRENT_LINK_FLAGS -Qopenmp)
    # Flags to enable offload of DO CONCURRENT
    set(OpenMP_OFFLOAD_FLAGS -Qopenmp-targets=spir64)
    set(DO_CONCURRENT_OFFLOAD_COMPILE_FLAGS
        ${OpenMP_OFFLOAD_FLAGS}
        -switch:offload_do_concurrent)
    set(DO_CONCURRENT_OFFLOAD_LINK_FLAGS -Qiopenmp ${OpenMP_OFFLOAD_FLAGS})
else()
    # Flags to enable parallelization of DO CONCURRENT
    # In CMake 3.25.0+ adding OpenMP will make adding the flag unnecessary
    set(DO_CONCURRENT_LINK_FLAGS -fiopenmp)
    # CMake already passes -qopenmp during linking, but ifx expects to see
    # -fopenmp with -fopenmp-targets=
    set(OpenMP_OFFLOAD_FLAGS -fopenmp-targets=spir64)
    set(DO_CONCURRENT_OFFLOAD_COMPILE_FLAGS
        ${OpenMP_OFFLOAD_FLAGS}
        -switch offload_do_concurrent)
    set(DO_CONCURRENT_OFFLOAD_LINK_FLAGS -fiopenmp ${OpenMP_OFFLOAD_FLAGS})
endif()

add_executable(mandelbrot-cpu mandelbrot.f90)
target_link_libraries(mandelbrot-cpu PUBLIC OpenMP::OpenMP_Fortran pnm)
target_compile_options(mandelbrot-cpu PRIVATE ${OPT_FLAGS})
target_link_options(mandelbrot-cpu PRIVATE ${DO_CONCURRENT_LINK_FLAGS})

# The GPU version
# In addition to the OpenMP target offload flags, add DO CONCURRENT offload flag
add_executable(mandelbrot-gpu mandelbrot.f90)
target_link_libraries(mandelbrot-gpu PUBLIC OpenMP::OpenMP_Fortran pnm)
target_compile_options(mandelbrot-gpu PRIVATE ${DO_CONCURRENT_OFFLOAD_COMPILE_FLAGS})
target_link_options(mandelbrot-gpu PRIVATE ${DO_CONCURRENT_OFFLOAD_LINK_FLAGS})

