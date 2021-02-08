cmake_minimum_required(VERSION 3.5)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

project(LidarObjectDetection-PointPillars VERSION 1.0.0)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_COMPILER dpcpp)

############################################################
# gather all dependencies
############################################################
find_package(InferenceEngine REQUIRED)
find_path(SYCL_HEADERS CL/sycl.hpp PATHS $ENV{ONEAPI_ROOT}/compiler/latest/linux/include/sycl PATH_SUFFIXES / NO_CMAKE_FIND_ROOT_PATH NO_CMAKE_SYSTEM_PATH NO_DEFAULT_PATH)
find_path(DPCPP_HEADERS oneapi/dpl/iterator PATHS $ENV{ONEAPI_ROOT}/dpl/latest/linux/include PATH_SUFFIXES / NO_CMAKE_FIND_ROOT_PATH NO_CMAKE_SYSTEM_PATH NO_DEFAULT_PATH)
find_path(DPCT_HEADERS dpct/dpct.hpp PATHS $ENV{ONEAPI_ROOT}/dpcpp-ct/latest/include PATH_SUFFIXES / NO_CMAKE_FIND_ROOT_PATH NO_CMAKE_SYSTEM_PATH NO_DEFAULT_PATH)
find_library(SYCL_LIB sycl PATHS $ENV{ONEAPI_ROOT}/compiler/latest/linux PATH_SUFFIXES lib NO_CMAKE_FIND_ROOT_PATH NO_CMAKE_SYSTEM_PATH NO_DEFAULT_PATH)

find_package(TBB REQUIRED)

if(TBB_FOUND)
  message(STATUS "TBB Version - ${TBB_VERSION}")
endif()

if(NOT DPCPP_HEADERS)
  message(FATAL_ERROR "No DPCPP Headers for oneAPI found. Please check the environment variable ONEAPI_ROOT")
else()
  message(STATUS "DPCPP Headers - ${DPCPP_HEADERS}")
endif()

if(NOT DPCT_HEADERS)
  message(FATAL_ERROR "No DPCT Headers for oneAPI found. Please check the environment variable ONEAPI_ROOT")
else()
  message(STATUS "DPCT Headers - ${DPCT_HEADERS}")
endif()

if(NOT SYCL_HEADERS)
  message(FATAL_ERROR "No SYCL Headers for oneAPI found. Please check the environment variable ONEAPI_ROOT")
else()
  message(STATUS "SYCL Headers - ${SYCL_HEADERS}")
endif()

if(NOT SYCL_LIB)
  message(FATAL_ERROR "No SYCL Library for oneAPI found. Please check the environment variable ONEAPI_ROOT")
else()
  message(STATUS "SYCL Library - ${SYCL_LIB}")
endif()


############################################
# Build Library
############################################

set(PROJECT_SOURCE_LIST
  ${CMAKE_SOURCE_DIR}/src/PointPillars/operations/anchorgrid.cpp
  ${CMAKE_SOURCE_DIR}/src/PointPillars/operations/scan.cpp
  ${CMAKE_SOURCE_DIR}/src/PointPillars/operations/scatter.cpp
  ${CMAKE_SOURCE_DIR}/src/PointPillars/operations/nms.cpp
  ${CMAKE_SOURCE_DIR}/src/PointPillars/operations/postprocess.cpp
  ${CMAKE_SOURCE_DIR}/src/PointPillars/operations/maskanchors.cpp
  ${CMAKE_SOURCE_DIR}/src/PointPillars/operations/preprocess.cpp
  ${CMAKE_SOURCE_DIR}/src/PointPillars/inference/pointpillars.cpp
)

add_library(${PROJECT_NAME} SHARED
            ${PROJECT_SOURCE_LIST}
            )

target_link_libraries(${PROJECT_NAME} PUBLIC
                      ${InferenceEngine_LIBRARIES}
                      dl
                      ${TBB_LIBRARIES}
                      ${SYCL_LIB}
                    )

target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
  ${SYCL_HEADERS}
  ${DPCPP_HEADERS}
  ${DPCT_HEADERS}
  ${TBB_INCLUDE_DIRS}
  ${InferenceEngine_INCLUDE_DIRS}
)

target_include_directories(${PROJECT_NAME} PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(${PROJECT_NAME} PRIVATE -fsycl -Wall -Wextra -Wformat-security)

target_compile_definitions(${PROJECT_NAME} PUBLIC -DCL_TARGET_OPENCL_VERSION=220)



############################################
# Build Example
############################################

find_package(Boost REQUIRED COMPONENTS filesystem program_options)

add_executable(example.exe src/main.cpp)

target_link_libraries(example.exe PRIVATE
  ${PROJECT_NAME}
  Boost::filesystem
  Boost::program_options
)

configure_file(${CMAKE_SOURCE_DIR}/data/model/pfe.onnx ${CMAKE_CURRENT_BINARY_DIR}/pfe.onnx COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/data/model/rpn.onnx ${CMAKE_CURRENT_BINARY_DIR}/rpn.onnx COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/data/example.pcd ${CMAKE_CURRENT_BINARY_DIR}/example.pcd COPYONLY)

