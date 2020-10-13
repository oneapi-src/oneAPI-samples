# required cmake version
cmake_minimum_required(VERSION 3.5)

project (dpc_reduce)
set(CMAKE_CXX_COMPILER "mpiicpc")

# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()


# For beta10 and later due to change in location for dpl library.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsycl -std=c++17 -I $ENV{ONEAPI_ROOT}/dpl/2021.1-beta10/linux/include -fsycl-unnamed-lambda")

# Beta 09
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsycl -std=c++17 -fsycl-unnamed-lambda")


set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ltbb -lsycl")

add_executable (dpc_reduce src/main.cpp)

add_custom_target (run
        COMMAND CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE=16MB  ./dpc_reduce
        WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)
