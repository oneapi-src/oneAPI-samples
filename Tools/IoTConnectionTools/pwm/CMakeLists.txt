cmake_minimum_required (VERSION 3.0)

# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

# Set language standard
set(CMAKE_CXX_STANDARD "11")

project (PWM)

# OS checks
if(WIN32)
    message(FATAL_ERROR, "This sample cannot be compiled natively on Windows OS.")
endif()

# Dependency checks
find_package(PkgConfig)
pkg_search_module (MRAA REQUIRED mraa)

add_subdirectory(cpp)
