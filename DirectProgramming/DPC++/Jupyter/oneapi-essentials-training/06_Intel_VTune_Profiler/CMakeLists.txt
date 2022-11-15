# CMakeLists.txt for ISO3DFD_SYCL project
project (iso3dfd_sycl)

# required cmake version
cmake_minimum_required(VERSION 3.5)

OPTION(SHARED_KERNEL "Use SLM Kernel Version - Only for GPU" OFF)

set(CMAKE_CXX_COMPILER "icpx")

if(SHARED_KERNEL)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DUSE_SHARED -fsycl -std=c++11")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsycl -std=c++11")
endif(SHARED_KERNEL)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lOpenCL -lsycl")

add_executable (iso3dfd src/iso3dfd.cpp src/iso3dfd_kernels_cpu.cpp src/iso3dfd_kernels_gpu.cpp src/utils.cpp)

add_custom_target (run 
	COMMAND iso3dfd 256 256 256 256 8 8 10 sycl gpu
	WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

add_custom_target (run_cpu 
	COMMAND iso3dfd 256 256 256 256 8 8 10 sycl cpu
	WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

