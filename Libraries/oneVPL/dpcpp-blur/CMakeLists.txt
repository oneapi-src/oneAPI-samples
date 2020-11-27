cmake_minimum_required(VERSION 3.10.2)
# This sample must be compiled and executed within the oneAPI product enviroment
# with the dpcpp compiler to use dpcpp features
set(DPCPP_COMPILER_NAME "dpcpp")
find_program(DPCPP_COMPILER ${DPCPP_COMPILER_NAME})

if(DPCPP_COMPILER)
  set(CMAKE_CXX_COMPILER ${DPCPP_COMPILER_NAME})
  message(STATUS "DPCPP compiler found, blur feature enabled")
else()
  message(STATUS "DPCPP compiler not found, blur feature disabled")
endif()

project(dpcpp-blur LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set(TARGET dpcpp-blur)
set(SOURCES src/dpcpp-blur.cpp)
set(RUNARGS ${CMAKE_CURRENT_SOURCE_DIR}/content/cars_128x96.i420 128 96)

# Set default build type to RelWithDebInfo if not specified
if(NOT CMAKE_BUILD_TYPE)
  message(
    STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
  set(CMAKE_BUILD_TYPE
      "RelWithDebInfo"
      CACHE
        STRING
        "Choose build type from: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

add_executable(${TARGET} ${SOURCES})

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  if(NOT DEFINED ENV{VSCMD_VER})
    set(CMAKE_MSVCIDE_RUN_PATH $ENV{PATH})
  endif()
endif()

if(POLICY CMP0074)
  # ignore warning of VPL_ROOT in find_package search path
  cmake_policy(SET CMP0074 OLD)
endif()

if(DPCPP_COMPILER)
  add_definitions(-DBUILD_DPCPP)
  target_link_libraries(${TARGET} OpenCL sycl)
endif()

if(TARGET VPL)
  target_link_libraries(${TARGET} VPL)
else()
  find_package(VPL REQUIRED)
  target_link_libraries(${TARGET} VPL::dispatcher)
endif()

get_directory_property(has_parent PARENT_DIRECTORY)
if(NOT has_parent)
  # only make run target available for stand-alone build
  add_custom_target(run ${TARGET} ${RUNARGS})
endif()
