cmake_minimum_required(VERSION 3.10.2)
project(hello-encode)

set(TARGET hello-encode)
set(SOURCES src/hello-encode.cpp)
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
