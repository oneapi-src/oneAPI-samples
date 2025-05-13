############################################################################
## Copyright © 2025 Codeplay Software
##
## SPDX-License-Identifier: MIT
############################################################################

cmake_minimum_required(VERSION 3.12)

if (NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER icpx)
endif()

project(SYCL-Graph-Samples)

# Set global flags
set(CMAKE_CXX_STANDARD 17)

# Configure SYCL
include("${CMAKE_SOURCE_DIR}/../common/cmake/ConfigureSYCL.cmake")

# Output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} CACHE PATH "" FORCE)

# Find all .cpp files in the src/ directory
file(GLOB SAMPLE_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

# Add executable for each .cpp file
foreach(SOURCE_FILE ${SAMPLE_SOURCES})
    # Extract the file name without the extension
    get_filename_component(EXE_NAME ${SOURCE_FILE} NAME_WE)

    # Create executable
    add_executable(${EXE_NAME} ${SOURCE_FILE})

    # Add SYCL flags
  target_compile_options(${EXE_NAME} PUBLIC ${SYCL_FLAGS})
  target_link_options(${EXE_NAME} PUBLIC ${SYCL_FLAGS})
endforeach()
