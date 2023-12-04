cmake_minimum_required(VERSION 3.5)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl") 

project(LidarObjectDetection-PointPillars VERSION 1.0.0)

############################################################
# gather all dependencies
############################################################
find_package(OpenVINO REQUIRED)

find_path(SYCL_HEADERS CL/sycl.hpp PATHS $ENV{ONEAPI_ROOT}/compiler/latest/include/sycl PATH_SUFFIXES / NO_CMAKE_FIND_ROOT_PATH NO_CMAKE_SYSTEM_PATH NO_DEFAULT_PATH)
find_path(DPCPP_HEADERS oneapi/dpl/iterator PATHS $ENV{ONEAPI_ROOT}/dpl/latest/include PATH_SUFFIXES / NO_CMAKE_FIND_ROOT_PATH NO_CMAKE_SYSTEM_PATH NO_DEFAULT_PATH)
find_library(SYCL_LIB sycl PATHS $ENV{ONEAPI_ROOT}/compiler/latest PATH_SUFFIXES lib NO_CMAKE_FIND_ROOT_PATH NO_CMAKE_SYSTEM_PATH NO_DEFAULT_PATH)

if(NOT DPCPP_HEADERS)
  message(FATAL_ERROR "No DPCPP Headers for oneAPI found. Please check the environment variable ONEAPI_ROOT")
else()
  message(STATUS "DPCPP Headers - ${DPCPP_HEADERS}")
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

# OpenCL dependency
find_package(OpenCLHeaders REQUIRED)
find_package(OpenCLHeadersCpp REQUIRED)
find_package(OpenCLICDLoader REQUIRED)

############################################
# Build Library
############################################

set(PROJECT_SOURCE_LIST
  ${CMAKE_SOURCE_DIR}/src/pointpillars/anchorgrid.cpp
  ${CMAKE_SOURCE_DIR}/src/pointpillars/scan.cpp
  ${CMAKE_SOURCE_DIR}/src/pointpillars/scatter.cpp
  ${CMAKE_SOURCE_DIR}/src/pointpillars/nms.cpp
  ${CMAKE_SOURCE_DIR}/src/pointpillars/postprocess.cpp
  ${CMAKE_SOURCE_DIR}/src/pointpillars/preprocess.cpp
  ${CMAKE_SOURCE_DIR}/src/pointpillars/pointpillars.cpp
)

add_library(${PROJECT_NAME} SHARED
            ${PROJECT_SOURCE_LIST}
            )

target_link_libraries(${PROJECT_NAME} PUBLIC
  OpenCL::Headers
  OpenCL::HeadersCpp
  openvino::runtime
  dl
  ${SYCL_LIB}
)

target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
  ${SYCL_HEADERS}
  ${DPCPP_HEADERS}
  ${InferenceEngine_INCLUDE_DIRS}
)

target_include_directories(${PROJECT_NAME} PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(${PROJECT_NAME} PRIVATE -fsycl)

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
  OpenCL::OpenCL
)

# Get the required model files and copy the input point cloud
message(STATUS "Getting model files")
execute_process(COMMAND wget https://github.com/k0suke-murakami/kitti_pretrained_point_pillars/raw/master/pfe.onnx -q -O ${CMAKE_CURRENT_BINARY_DIR}/pfe.onnx)
execute_process(COMMAND wget https://github.com/k0suke-murakami/kitti_pretrained_point_pillars/raw/master/rpn.onnx -q -O ${CMAKE_CURRENT_BINARY_DIR}/rpn.onnx)
configure_file(${CMAKE_SOURCE_DIR}/data/example.pcd ${CMAKE_CURRENT_BINARY_DIR}/example.pcd COPYONLY)

