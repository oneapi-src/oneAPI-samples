cmake_minimum_required (VERSION 3.0)
project (HELLO-IOT-WORLD CXX)

# Use Intel C++ Compiler Classic when available
find_program(ICPC NAMES icpc icl)
if(ICPC)
    set(CMAKE_CXX_COMPILER ${ICPC})
    message(STATUS "Switching to the Intel(R) C++ Compiler Classic from: ${ICPC}")
else()
    message("-- Could not find the Intel(R) C++ Compiler Classic on the system path. Did you setup the compiler environment?")
endif()

# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

# Set language standard
set(CMAKE_CXX_STANDARD "11")

add_subdirectory(cpp)
