cmake_minimum_required(VERSION 3.16)
project(ospTutorial LANGUAGES C CXX)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

list(APPEND CMAKE_PREFIX_PATH "${ONEAPI_ROOT}/rkcommon/latest")
find_package(ospray REQUIRED)

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
include_directories(${OSPRAY_INCLUDE_DIR} ${RKCOMMON_INCLUDE_DIRS})
link_directories(${OSPRAY_ROOT}/lib ${ONEAPI_ROOT}/rkcommon/latest/lib)

add_executable(${PROJECT_NAME} ${OSPRAY_RESOURCE} src/ospTutorial.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ospray rkcommon)

install(TARGETS ${PROJECT_NAME}
  DESTINATION ${CMAKE_INSTALL_BINDIR}
)
