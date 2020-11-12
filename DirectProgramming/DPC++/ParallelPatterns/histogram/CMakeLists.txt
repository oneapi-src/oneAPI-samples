cmake_minimum_required (VERSION 3.5)
project(histogram LANGUAGES CXX)

set(CMAKE_CXX_COMPILER "dpcpp")

# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release")
    set (CMAKE_BUILD_TYPE "Release" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

add_executable (histogram src/main.cpp)

add_custom_target (run
        COMMAND ./histogram
        WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)


