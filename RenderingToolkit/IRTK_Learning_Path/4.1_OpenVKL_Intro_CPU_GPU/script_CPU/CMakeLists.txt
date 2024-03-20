## Copyright 2022 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

# Build using
# source /opt/intel/oneapi/setvars.sh 
# cmake ..
# 
# Run With:
# source /opt/intel/oneapi/setvars.sh
# ./vklMinimal_CPU_<number>

cmake_minimum_required(VERSION 3.1)
project(OPENVKL_CPU_INTRO LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../bin)

find_package(openvkl 2.0.0 CONFIG REQUIRED)

find_package(rkcommon REQUIRED)

# The minimal_01 ... minimal_06 examples gradually increase in complexity
foreach(i 01 02 03 04 05 06)
  add_executable(vklMinimal_CPU_${i} ../src/minimal_${i}.cpp ${VKL_RESOURCE})
  target_link_libraries(vklMinimal_CPU_${i} PRIVATE openvkl::openvkl openvkl::openvkl_module_cpu_device rkcommon::rkcommon)
  install(TARGETS vklMinimal_CPU_${i} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endforeach()
