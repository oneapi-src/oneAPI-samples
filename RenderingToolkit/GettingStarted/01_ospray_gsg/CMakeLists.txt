cmake_minimum_required(VERSION 3.16)
project(OSPRAY_GSG LANGUAGES CXX)

set(ONEAPI_ROOT "")
if(DEFINED ENV{ONEAPI_ROOT})
  set(ONEAPI_ROOT "$ENV{ONEAPI_ROOT}")
  message(STATUS "ONEAPI_ROOT FROM ENVIRONMENT: ${ONEAPI_ROOT}")
else()
  message(FATAL_ERROR "ONEAPI_ROOT NOT set. Please source environment variables.")
endif(DEFINED ENV{ONEAPI_ROOT})

set(RKCOMMON_BASE_PATH "")
set(OSPRAY_BASE_PATH "")
if(EXISTS ${ONEAPI_ROOT}/oneapi-vars.sh OR EXISTS ${ONEAPI_ROOT}/oneapi-bars.bat)
  set(RKCOMMON_BASE_PATH "${ONEAPI_ROOT}")
  set(OSPRAY_BASE_PATH "${ONEAPI_ROOT}")
else()
  set(RKCOMMON_BASE_PATH "${ONEAPI_ROOT}/rkcommon/latest")
  set(OSPRAY_BASE_PATH "${ONEAPI_ROOT}/ospray/latest")
endif(EXISTS ${ONEAPI_ROOT}/oneapi-vars.sh OR EXISTS ${ONEAPI_ROOT}/oneapi-bars.bat)

list(APPEND CMAKE_PREFIX_PATH "${RKCOMMON_BASE_PATH}")
find_package(ospray REQUIRED PATHS ${OSPRAY_BASE_PATH})

if(MSVC)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif(MSVC)

include_directories(${OSPRAY_INCLUDE_DIR} ${RKCOMMON_INCLUDE_DIRS})
link_directories(${OSPRAY_ROOT}/lib ${RKCOMMON_BASE_PATH}/lib)

add_executable(ospTutorialCpp ${OSPRAY_RESOURCE} src/ospTutorial.cpp)
target_link_libraries(ospTutorialCpp PRIVATE ospray rkcommon)

install(TARGETS ospTutorialCpp
  DESTINATION ${CMAKE_INSTALL_BINDIR}
)
