cmake_minimum_required(VERSION 3.16)
project(embree_gsg LANGUAGES C CXX)

find_package(embree 3.0 REQUIRED)

set(ONEAPI_ROOT "")
if($ENV{ONEAPI_ROOT})
	set(ONEAPI_ROOT $ENV{ONEAPI_ROOT})
else()
	if(WIN32)
		set(ONEAPI_ROOT "C:/Program Files (x86)/Intel/oneAPI")
	else()
		set(ONEAPI_ROOT /opt/intel/oneapi)
	endif()
endif($ENV{ONEAPI_ROOT})


message(STATUS "ONEAPI_ROOT: ${ONEAPI_ROOT}")
include_directories(${EMBREE_INCLUDE_DIR} ${RKCOMMON_INCLUDE_DIRS})
link_directories(${EMBREE_ROOT}/lib ${ONEAPI_ROOT}/rkcommon/latest/lib)

add_executable(minimal src/minimal.cpp)
target_link_libraries(minimal PRIVATE embree rkcommon)

install(TARGETS minimal
  DESTINATION ${CMAKE_INSTALL_BINDIR}
)
