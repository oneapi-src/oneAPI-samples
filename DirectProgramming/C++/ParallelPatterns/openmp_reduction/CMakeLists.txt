set(CMAKE_CXX_COMPILER "icpx")
# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release")
    set (CMAKE_BUILD_TYPE "Release" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

cmake_minimum_required (VERSION 3.0)
project(openmp_reduction LANGUAGES CXX)
add_subdirectory (src)
