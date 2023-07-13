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

# A simple library to generate images for Mandlebrot
add_library(pnm-omp STATIC pnm-omp.f90)

# The GNU (Linux and Mac) and MSVC (Windows) frontend variants spell options
# differently
if(CMAKE_Fortran_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(openmp_target_offload_flags -Qopenmp -Qopenmp-targets=spir64)
else()
    set(openmp_target_offload_flags -fiopenmp -fopenmp-targets=spir64)
endif()

# The "Hello, world" application compiled for host
add_executable(hello-omp-fortran hello_omp.f90)
target_link_libraries(hello-omp-fortran PUBLIC OpenMP::OpenMP_Fortran)

# "Hello, world" for GPU offload
add_executable(hello-omp-gpu-fortran hello_omp_gpu.f90)
target_link_libraries(hello-omp-gpu-fortran PUBLIC OpenMP::OpenMP_Fortran)
target_compile_options(hello-omp-gpu-fortran PRIVATE ${openmp_target_offload_flags})
target_link_options(hello-omp-gpu-fortran PRIVATE ${openmp_target_offload_flags})

# OpenMP mandlebrot program for host
add_executable(mandelbrot-cpu-omp mandelbrot-omp.f90)
target_link_libraries(mandelbrot-cpu-omp PUBLIC OpenMP::OpenMP_Fortran pnm-omp)
target_compile_options(mandelbrot-cpu-omp PRIVATE)

# OpenMP mandlebrot program for GPU
add_executable(mandelbrot-gpu-omp mandelbrot-omp.f90)
target_link_libraries(mandelbrot-gpu-omp PUBLIC OpenMP::OpenMP_Fortran pnm-omp)
target_compile_options(mandelbrot-gpu-omp PRIVATE ${openmp_target_offload_flags})
target_link_options(mandelbrot-gpu-omp PRIVATE ${openmp_target_offload_flags})
