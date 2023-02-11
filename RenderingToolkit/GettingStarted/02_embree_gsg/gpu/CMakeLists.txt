## Copyright 2009-2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

project(EMBREE_SYCL_GSG LANGUAGES CXX)
cmake_minimum_required(VERSION 3.2.0)


set(ONEAPI_ROOT "")
if (NOT "$ENV{ONEAPI_ROOT}")
	set(ONEAPI_ROOT /opt/intel/oneapi)
else()
	set(ONEAPI_ROOT $ENV{ONEAPI_ROOT})

endif()


#Find embree4
find_package(embree 4.0 REQUIRED HINTS ${ONEAPI_ROOT}/embree/latest)
#Find compiler toolchain for DPCPP SYCL
find_package(IntelDPCPP REQUIRED)

#Required for SYCL implementation support
set(CMAKE_CXX_STANDARD 17)



  # This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

  # JIT mode compile flags
    set(SYCL_COMPILE_FLAGS "-fsycl -Xclang -fsycl-allow-func-ptr -fsycl-targets=spir64 ${WIN_FLAG}")
  set(SYCL_LINK_FLAGS "")
    if(NOT MSVC)
    set(SYCL_LINK_FLAGS "-fsycl -fsycl-targets=spir64")
    endif(NOT MSVC)


  GET_FILENAME_COMPONENT(SYCL_COMPILER_DIR ${CMAKE_CXX_COMPILER} PATH)
 include_directories(SYSTEM "${SYCL_COMPILER_DIR}/../include/sycl" "${SYCL_COMPILER_DIR}/../include")

  ADD_EXECUTABLE(minimal_sycl src/minimal_sycl.cpp)
  TARGET_LINK_LIBRARIES(minimal_sycl embree)
set_property(TARGET minimal_sycl APPEND PROPERTY COMPILE_FLAGS ${SYCL_COMPILE_FLAGS})
set_property(TARGET minimal_sycl APPEND PROPERTY LINK_FLAGS    ${SYCL_LINK_FLAGS})

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
endif()

install(TARGETS minimal_sycl
        RUNTIME DESTINATION bin/)

