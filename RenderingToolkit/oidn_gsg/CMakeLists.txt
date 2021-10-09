cmake_minimum_required(VERSION 3.16)
project(OIDN_GSG LANGUAGES C CXX)

find_package(OpenImageDenoise REQUIRED)
find_package(TBB REQUIRED)

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
set(OIDN_ROOT ${ONEAPI_ROOT}/oidn/latest)
set(OIDN_INCLUDE_DIR ${OIDN_ROOT}/include)
set(OIDN_LIB_DIR ${OIDN_ROOT}/lib)
message(STATUS "ONEAPI_ROOT: ${ONEAPI_ROOT}")

add_subdirectory(src/common)
add_subdirectory(src/apps/utils)

include_directories(${OIDN_INCLUDE_DIR} ${OIDN_ROOT} ${OIDN_GSG_SOURCE_DIR}/src)
link_directories(${OIDN_ROOT}/lib)

add_executable(oidnDenoise src/apps/oidnDenoise.cpp)
target_link_libraries(oidnDenoise PRIVATE common utils OpenImageDenoise)

install(TARGETS oidnDenoise
  DESTINATION ${CMAKE_INSTALL_BINDIR}
)
