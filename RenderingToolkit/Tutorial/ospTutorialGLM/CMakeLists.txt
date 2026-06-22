cmake_minimum_required(VERSION 3.16)
project(OSPRAY_TUTORIAL_GLM LANGUAGES CXX)

include(CMakePrintHelpers)
include(GNUInstallDirs)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH
  ${PROJECT_SOURCE_DIR}/cmake/dependencies
)


set(INSTALL_DIR_ABSOLUTE "Release")
set(ONEAPI_ROOT "")
if(DEFINED ENV{ONEAPI_ROOT})
  set(ONEAPI_ROOT "$ENV{ONEAPI_ROOT}")
  message(STATUS "ONEAPI_ROOT FROM ENVIRONMENT: ${ONEAPI_ROOT}")
else()
  if(WIN32)
    set(ONEAPI_ROOT "C:/Program Files (x86)/Intel/oneAPI")
  else()
    set(ONEAPI_ROOT /opt/intel/oneapi)
  endif()
  message(STATUS "ONEAPI_ROOT DEFAULT: ${ONEAPI_ROOT}")
endif(DEFINED ENV{ONEAPI_ROOT})

list(APPEND CMAKE_PREFIX_PATH "${ONEAPI_ROOT}/rkcommon/latest")
find_package(ospray REQUIRED PATHS ${ONEAPI_ROOT})

if(MSVC)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif(MSVC)

include_directories(${OSPRAY_INCLUDE_DIR} ${RKCOMMON_INCLUDE_DIRS})
link_directories(${OSPRAY_ROOT}/lib ${ONEAPI_ROOT}/rkcommon/latest/lib)

include(cmake/ospray_macros.cmake)
include(cmake/dependencies/glm.cmake)

link_libraries(ospray rkcommon)

add_executable(ospTutorialGLM ${OSPRAY_RESOURCE} ospTutorialGLM.cpp)
target_link_libraries(ospTutorialGLM PRIVATE ospray glm)
