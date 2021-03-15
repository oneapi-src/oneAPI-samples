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
find_package(InferenceEngine)

if (InferenceEngine_FOUND)
  FILE(STRINGS "${InferenceEngine_INCLUDE_DIRS}/../version.txt" IE_VERSION)
  if (${IE_VERSION} STRLESS "2021.2")
    unset(InferenceEngine_FOUND)
  endif()
endif()

if (NOT InferenceEngine_FOUND)
  message(STATUS "ERROR: OpenVINO with version 2021.2 or newer required.")
  if (IE_VERSION)
    message(STATUS "   Found only ${IE_VERSION}")
  endif()
  message(STATUS "  Please ensure that OpenVINO is correctly installed and that the OpenVINO environment is setup,")
  message(STATUS "  i.e. run 'source <path/to/openvino>/bin/setupvars.sh' before calling cmake")
  return()
endif()


find_path(SYCL_HEADERS CL/sycl.hpp PATHS $ENV{ONEAPI_ROOT}/compiler/latest/linux/include/sycl PATH_SUFFIXES / NO_CMAKE_FIND_ROOT_PATH NO_CMAKE_SYSTEM_PATH NO_DEFAULT_PATH)
find_path(DPCPP_HEADERS oneapi/dpl/iterator PATHS $ENV{ONEAPI_ROOT}/dpl/latest/linux/include PATH_SUFFIXES / NO_CMAKE_FIND_ROOT_PATH NO_CMAKE_SYSTEM_PATH NO_DEFAULT_PATH)
find_library(SYCL_LIB sycl PATHS $ENV{ONEAPI_ROOT}/compiler/latest/linux PATH_SUFFIXES lib NO_CMAKE_FIND_ROOT_PATH NO_CMAKE_SYSTEM_PATH NO_DEFAULT_PATH)

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
                      ${InferenceEngine_LIBRARIES}
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
)

# Get the required model files and copy the input point cloud
message(STATUS "Getting model files")
execute_process(COMMAND wget https://github.com/k0suke-murakami/kitti_pretrained_point_pillars/raw/master/pfe.onnx -q -O ${CMAKE_CURRENT_BINARY_DIR}/pfe.onnx)
execute_process(COMMAND wget https://github.com/k0suke-murakami/kitti_pretrained_point_pillars/raw/master/rpn.onnx -q -O ${CMAKE_CURRENT_BINARY_DIR}/rpn.onnx)
configure_file(${CMAKE_SOURCE_DIR}/data/example.pcd ${CMAKE_CURRENT_BINARY_DIR}/example.pcd COPYONLY)

