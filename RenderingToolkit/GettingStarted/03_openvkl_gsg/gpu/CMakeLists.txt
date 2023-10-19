cmake_minimum_required(VERSION 3.25)
project(OPENVKL_GPU_GSG LANGUAGES CXX)

set(ONEAPI_ROOT "")
if(DEFINED ENV{ONEAPI_ROOT})
  set(ONEAPI_ROOT "$ENV{ONEAPI_ROOT}")
  message(STATUS "ONEAPI_ROOT FROM ENVIRONMENT: ${ONEAPI_ROOT}")
else()
  message(FATAL_ERROR "ONEAPI_ROOT NOT set. Please source environment variables.")
endif(DEFINED ENV{ONEAPI_ROOT})

set(OPENVKL_BASE_DIR "")
set(COMPILER_SYCL_INCLUDE "")
set(COMPILER_CL_INCLUDE "")
set(COMPILER_PATH "")
if(EXISTS ${ONEAPI_ROOT}/oneapi-vars.sh OR EXISTS ${ONEAPI_ROOT}/oneapi-bars.bat)
  set(OPENVKL_BASE_DIR ${ONEAPI_ROOT})
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

set(DEFAULT_COMPILER ${COMPILER_PATH}/clang++)
if(NOT DEFINED CMAKE_CXX_COMPILER)
	message(STATUS "CMAKE_CXX_COMPILER not set. using ${DEFAULT_COMPILER}")
	set(CMAKE_CXX_COMPILER ${DEFAULT_COMPILER})
endif(NOT DEFINED CMAKE_CXX_COMPILER)

set(OPENVKL_INCLUDE_DIR ${OPENVKL_BASE_DIR}/include)
#target headers
set(OPENVKL_GPU_INCLUDE_DIR ${OPENVKL_INCLUDE_DIR}/openvkl/devices/gpu)

find_package(openvkl REQUIRED PATHS ${OPENVKL_BASE_DIR})

if(MSVC)
	#  set(CMAKE_CXX_STANDARD 11)
	#  set(CMAKE_CXX_STANDARD_REQUIRED ON)
	#  set(CMAKE_CXX_EXTENSIONS OFF)
	#set(CMAKE_CXX_FLAGS "-opaque-pointers")
	  set (CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld-link")

else()
	#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif(MSVC)

#include_directories(${OPENVKL_INCLUDE_DIR} ${OPENVKL_GPU_INCLUDE_DIR} ${COMPILER_SYCL_INCLUDE} ${COMPILER_CL_INCLUDE})
#include_directories(${OPENVKL_INCLUDE_DIR} ${OPENVKL_GPU_INCLUDE_DIR} /home/mcarroll/sycl-nightly-20230922/include/sycl /home/mcarroll/sycl-nightly-20230922/include/sycl/sycl)
#include_directories(${OPENVKL_INCLUDE_DIR} ${OPENVKL_GPU_INCLUDE_DIR})
link_directories(${OPENVKL_BASE_DIR}/lib)

add_executable(vklTutorialGPU src/vklTutorialGPU.cpp)
target_link_libraries(vklTutorialGPU PRIVATE openvkl::openvkl openvkl::openvkl_module_gpu_device)
install(TARGETS vklTutorialGPU RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
