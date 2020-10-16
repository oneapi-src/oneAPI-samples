cmake_minimum_required (VERSION 3.5)
project (all-pairs-shortest-paths)

set(CMAKE_CXX_COMPILER dpcpp)


# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++17")

add_executable(apsp src/apsp.cpp)

if(WIN32)
add_custom_target(run apsp.exe)
else()
add_custom_target(run ./apsp)
endif()