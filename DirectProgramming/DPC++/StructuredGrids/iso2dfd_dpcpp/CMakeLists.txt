# required cmake version
cmake_minimum_required(VERSION 3.5)

# CMakeLists.txt for ISO2DFD_DPCPP project
project (iso2dfd_dpcpp)

set(CMAKE_CXX_COMPILER "dpcpp")

# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsycl -std=c++17")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lOpenCL -lsycl")

add_executable (iso2dfd src/iso2dfd.cpp)

add_custom_target (run 
	COMMAND iso2dfd 1000 1000 2000 
	WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

