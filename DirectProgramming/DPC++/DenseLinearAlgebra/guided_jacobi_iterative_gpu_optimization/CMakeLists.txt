cmake_minimum_required (VERSION 3.5)

set(CMAKE_CXX_COMPILER dpcpp)

# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()
project(Jacobi_Iterative_Solver)

option(GPU_SELECTOR "Option description" OFF)
option(CPU_SELECTOR "Option description" OFF)

add_subdirectory (src)
