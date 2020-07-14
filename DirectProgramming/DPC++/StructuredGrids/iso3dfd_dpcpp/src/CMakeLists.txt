set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 --std=c++17")
include_directories("../include/")

OPTION(SHARED_KERNEL "Use SLM Kernel Version - Only for GPU" OFF)
if(SHARED_KERNEL)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_SHARED")
endif(SHARED_KERNEL)

# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

add_executable (iso3dfd.exe iso3dfd.cpp iso3dfd_kernels.cpp utils.cpp)
target_link_libraries(iso3dfd.exe OpenCL sycl)
if(WIN32)
        add_custom_target (run iso3dfd.exe 256 256 256 32 8 64 10 gpu)
        add_custom_target (run_cpu iso3dfd.exe 256 256 256 256 1 1 10 cpu)
else()
        add_custom_target (run iso3dfd.exe 256 256 256 32 8 64 10 gpu)
        add_custom_target (run_cpu iso3dfd.exe 256 256 256 256 1 1 10 cpu)
endif()
