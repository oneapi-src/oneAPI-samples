cmake_minimum_required(VERSION 3.25)
project(OPENVKL_GPU_GSG LANGUAGES CXX)

set(ONEAPI_ROOT "")
if(DEFINED ENV{ONEAPI_ROOT})
  set(ONEAPI_ROOT "$ENV{ONEAPI_ROOT}")
  message(STATUS "ONEAPI_ROOT FROM ENVIRONMENT: ${ONEAPI_ROOT}")
else()
  message(FATAL_ERROR "ONEAPI_ROOT NOT set. Please source environment variables.")
endif(DEFINED ENV{ONEAPI_ROOT})

#lots of helper variables to get access to other components
set(OPENVKL_BASE_DIR "")
set(COMPILER_SYCL_INCLUDE "")
set(COMPILER_CL_INCLUDE "")
set(COMPILER_PATH "")
set(RKCOMMON_BASE_DIR "")
set(ISPCRT_BASE_DIR "")
set(TBB_BASE_DIR "")
set(EMBREE_BASE_DIR "")
if(EXISTS ${ONEAPI_ROOT}/oneapi-vars.sh OR EXISTS ${ONEAPI_ROOT}/oneapi-bars.bat)
  set(OPENVKL_BASE_DIR ${ONEAPI_ROOT})
  set(RKCOMMON_BASE_DIR ${ONEAPI_ROOT})
  set(ISPCRT_BASE_DIR ${ONEAPI_ROOT})
  set(TBB_BASE_DIR ${ONEAPI_ROOT})
  set(EMBREE_BASE_DIR ${ONEAPI_ROOT})
  if(MSVC)

  set(COMPILER_SYCL_INCLUDE ${ONEAPI_ROOT}/include)
  set(COMPILER_CL_INCLUDE ${ONEAPI_ROOT}/include/sycl)
  #  set(ONEAPI_INCLUDE_DIR ${ONEAPI_ROOT}/include 
  set(COMPILER_PATH ${ONEAPI_ROOT}/bin/compiler)
  else()
  set(COMPILER_SYCL_INCLUDE ${ONEAPI_ROOT}/opt/compiler/include/sycl)
  set(COMPILER_CL_INCLUDE ${ONEAPI_ROOT}/opt/compiler/include/sycl/sycl)
  #  set(ONEAPI_INCLUDE_DIR ${ONEAPI_ROOT}/include 
  set(COMPILER_PATH ${ONEAPI_ROOT}/bin/compiler)
  endif(MSVC)

else()
  set(OPENVKL_BASE_DIR ${ONEAPI_ROOT}/openvkl/latest)
  set(RKCOMMON_BASE_DIR ${ONEAPI_ROOT}/rkcommon/latest)
  set(ISPCRT_BASE_DIR ${ONEAPI_ROOT}/ispc/latest)
  set(TBB_BASE_DIR ${ONEAPI_ROOT}/tbb/latest)
  set(EMBREE_BASE_DIR ${ONEAPI_ROOT}/embree/latest)
  if(MSVC)
  set(COMPILER_SYCL_INCLUDE ${ONEAPI_ROOT}/compiler/latest/include)
  set(COMPILER_CL_INCLUDE ${ONEAPI_ROOT}/compiler/latest/include/sycl)
  set(COMPILER_PATH ${ONEAPI_ROOT}/compiler/latest/bin/compiler)
  else()
  set(COMPILER_SYCL_INCLUDE ${ONEAPI_ROOT}/compiler/latest/linux/include)
  set(COMPILER_CL_INCLUDE ${ONEAPI_ROOT}/compiler/latest/linux/include/sycl)
  set(COMPILER_PATH ${ONEAPI_ROOT}/compiler/latest/bin/compiler)
  endif(MSVC)
endif(EXISTS ${ONEAPI_ROOT}/oneapi-vars.sh OR EXISTS ${ONEAPI_ROOT}/oneapi-bars.bat)

if(NOT DEFINED CMAKE_CXX_COMPILER)
	message(FATAL_ERROR "CMAKE_CXX_COMPILER not set use icpx")
endif(NOT DEFINED CMAKE_CXX_COMPILER)

set(OPENVKL_INCLUDE_DIR ${OPENVKL_BASE_DIR}/include)
#target headers
set(OPENVKL_GPU_INCLUDE_DIR ${OPENVKL_INCLUDE_DIR}/openvkl/devices/gpu)

find_package(openvkl 2.0 REQUIRED PATHS ${OPENVKL_BASE_DIR})

find_package(rkcommon REQUIRED PATHS ${RKCOMMON_BASE_DIR})

find_package(TBB REQUIRED PATHS ${TBB_BASE_DIR}/lib/cmake/tbb)

find_package(embree 4.0 REQUIRED PATHS ${EMBREE_BASE_DIR})

link_directories(${OPENVKL_BASE_DIR}/lib ${RKCOMMON_BASE_DIR}/lib ${ISPCRT_BASE_DIR}/lib)

add_executable(vklTutorialGPU src/vklTutorialGPU.cpp)
target_link_libraries(vklTutorialGPU PRIVATE openvkl::openvkl openvkl::openvkl_module_gpu_device rkcommon ispcrt tbb embree)
install(TARGETS vklTutorialGPU RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
