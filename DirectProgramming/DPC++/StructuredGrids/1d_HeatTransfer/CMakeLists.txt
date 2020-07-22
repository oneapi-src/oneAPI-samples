# required cmake version
cmake_minimum_required(VERSION 3.5)

# CMakeLists.txt for 1d_HeatTransfer project
project (1d_HeatTransfer)

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

add_executable (1d_HeatTransfer src/1d_HeatTransfer.cpp)

add_custom_target (run 
	COMMAND 1d_HeatTransfer 100 1000 
	WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

