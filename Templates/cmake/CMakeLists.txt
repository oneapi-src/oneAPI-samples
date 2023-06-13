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

project(cmake-intel-examples)

set(projects
    OpenMP
    IPO
)

foreach(lang C CXX)
    foreach(project ${projects})
        add_subdirectory(${lang}/${project})
    endforeach()
endforeach()

add_subdirectory(Fortran/do_concurrent)
add_subdirectory(Fortran/OpenMP)
add_subdirectory(SYCL/add_sycl_to_target)
add_subdirectory(SYCL/target_link_library)
