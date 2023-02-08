cmake_minimum_required (VERSION 3.5)

set(CMAKE_CXX_COMPILER "icpx")

project(histogram LANGUAGES CXX)
# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release")
    set (CMAKE_BUILD_TYPE "Release" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++17 -fsycl")

add_executable (histogram src/main.cpp)

add_custom_target (run
        COMMAND ./histogram
        WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)


