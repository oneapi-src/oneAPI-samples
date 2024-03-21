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

add_library(common STATIC
  ${OIDN_INCLUDE_DIR}/OpenImageDenoise/config.h
  ${OIDN_INCLUDE_DIR}/OpenImageDenoise/oidn.h
  ${OIDN_INCLUDE_DIR}/OpenImageDenoise/oidn.hpp
  common.h
  common.cpp
  half.h
  half.cpp
  oidn_utils.h
  oidn_utils.cpp
  platform.h
  platform.cpp
  timer.h
)

target_include_directories(common
  PUBLIC
  ${OIDN_ROOT}
  ${OIDN_GSG_SOURCE_DIR}/src
)
