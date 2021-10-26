cmake_minimum_required(VERSION 3.16)
project(OSPRAY_GSG LANGUAGES CXX)

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
find_package(ospray REQUIRED PATHS ${ONEAPI_ROOT})
include_directories(${OSPRAY_INCLUDE_DIR} ${RKCOMMON_INCLUDE_DIRS})
link_directories(${OSPRAY_ROOT}/lib ${ONEAPI_ROOT}/rkcommon/latest/lib)

add_executable(ospTutorialCpp ${OSPRAY_RESOURCE} src/ospTutorial.cpp)
target_link_libraries(ospTutorialCpp PRIVATE ospray rkcommon)

install(TARGETS ospTutorialCpp
  DESTINATION ${CMAKE_INSTALL_BINDIR}
)
