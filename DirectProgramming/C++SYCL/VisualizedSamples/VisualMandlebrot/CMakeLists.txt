cmake_minimum_required (VERSION 3.4)

set(CMAKE_CXX_COMPILER icpx)

# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()
project (mandelbrot)

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17 -fsycl")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

add_executable(MandelbrotSDL MandelbrotSDL.cpp Mandel.hpp)
target_link_libraries(MandelbrotSDL OpenCL sycl ${SDL2_LIBRARIES})
add_custom_target(run ./MandelbrotSDL)
