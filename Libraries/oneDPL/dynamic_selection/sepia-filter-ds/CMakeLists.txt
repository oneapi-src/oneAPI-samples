set(CMAKE_CXX_COMPILER "icpx")
# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()
cmake_minimum_required (VERSION 3.4)

project (SepiaFilter)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsycl -std=c++17")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

add_executable(1_sepia_sycl src/1_sepia_sycl.cpp)
add_executable (2_sepia_policies src/2_sepia_policies.cpp)

target_link_libraries(1_sepia_sycl OpenCL sycl)
target_link_libraries(2_sepia_policies OpenCL sycl)

