cmake_minimum_required (VERSION 3.4.0)
if(NVIDIA_GPU)
	set(CMAKE_CXX_COMPILER "clang++")
else()
        set(CMAKE_CXX_COMPILER "icpx")
endif()
project (guided_simpleCUBLAS_LU_SYCLMigration)
# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
        message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
        set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
             STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
             FORCE)
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option (BUILD_SYCL_MIGRATED_SUBPROJECT "Build targets from subproject simpleCUBLAS_LU_migrated" ON)

if (BUILD_SYCL_MIGRATED_SUBPROJECT)
        add_subdirectory (02_sycl_migrated)
endif ()

