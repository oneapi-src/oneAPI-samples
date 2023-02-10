cmake_minimum_required(VERSION 3.16)
project(EMBREE_GSG LANGUAGES CXX)

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

find_package(embree 3.0 REQUIRED PATHS ${ONEAPI_ROOT})

if(MSVC)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif(MSVC)

include_directories(${EMBREE_INCLUDE_DIR} ${RKCOMMON_INCLUDE_DIRS})
link_directories(${EMBREE_ROOT}/lib ${ONEAPI_ROOT}/rkcommon/latest/lib)

add_executable(minimal src/minimal.cpp)
target_link_libraries(minimal PRIVATE embree rkcommon)

install(TARGETS minimal
  DESTINATION ${CMAKE_INSTALL_BINDIR}
)
