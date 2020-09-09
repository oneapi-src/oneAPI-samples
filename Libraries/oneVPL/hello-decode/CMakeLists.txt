cmake_minimum_required(VERSION 3.10.2)
project(hello-decode)

set(TARGET hello-decode)
set(SOURCES src/hello-decode.cpp)
set(VPL_NAME mfx)
set(CMAKE_MSVCIDE_RUN_PATH $ENV{PATH})
set(RUNARGS ${CMAKE_CURRENT_SOURCE_DIR}/content/cars_128x96.h265)

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

get_directory_property(has_parent PARENT_DIRECTORY)
if(NOT has_parent)
  # only make run target available for stand-alone build
  add_custom_target(run ${TARGET} ${RUNARGS})
endif()

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

if(TARGET mfx)
  target_link_libraries(${TARGET} mfx)
else()
  if(WIN32)
    set(VPL_NAME libmfx.lib)
  endif()
  message(STATUS "Check for required environment variables")
  if(NOT DEFINED ENV{VPL_LIB})
    message(FATAL_ERROR "Expected environment variable not defined: VPL_LIB")
  endif()
  if(NOT DEFINED ENV{VPL_INCLUDE})
    message(
      FATAL_ERROR "Expected environment variable not defined: VPL_INCLUDE")
  endif()

  message(STATUS "Check for presence of library")
  find_library(VPL_LIB ${VPL_NAME} HINTS $ENV{VPL_LIB})
  if(NOT VPL_LIB)
    message(FATAL_ERROR "vpl library not found. Looked in $ENV{VPL_LIB}")
  else()
    message(STATUS "Using vpl library: ${VPL_LIB}")
  endif()
  target_include_directories(${TARGET} PRIVATE $ENV{VPL_INCLUDE})
  target_link_libraries(${TARGET} ${VPL_LIB})
endif()
