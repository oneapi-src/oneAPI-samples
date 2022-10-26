set(CMAKE_CXX_COMPILER "dpcpp")
# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

cmake_minimum_required (VERSION 3.4)
project(gamma_correction LANGUAGES CXX)
add_subdirectory (src)