cmake_minimum_required(VERSION 3.16)
project(rkRayTracerGPU LANGUAGES CXX)

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

find_package(embree 4.0 REQUIRED PATHS ${ONEAPI_ROOT})

find_package(IntelDPCPP REQUIRED)

#Required for SYCL implementation support
set(CMAKE_CXX_STANDARD 17)

  # This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

GET_FILENAME_COMPONENT(SYCL_COMPILER_DIR ${CMAKE_CXX_COMPILER} PATH)

# Place holder for any needed compiler workarounds for std c++ version
# if(MSVC)
#  set(CMAKE_CXX_STANDARD_REQUIRED ON)
#  set(CMAKE_CXX_EXTENSIONS OFF)
#else()
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
#endif(MSVC)

# Set default build type to RelWithDebInfo if not specified
if(NOT MSVC)
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set. Using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)
endif(NOT MSVC)

# JIT mode compile flags
    set(SYCL_COMPILE_FLAGS "-fsycl -Xclang -fsycl-allow-func-ptr -fsycl-targets=spir64 ${WIN_FLAG}")
  set(SYCL_LINK_FLAGS "")
    if(NOT MSVC)
    set(SYCL_LINK_FLAGS "-fsycl -fsycl-targets=spir64")
    endif(NOT MSVC)


#stb headers are located in dev-utilities. oneAPI 2022.2 release and earlier: these header only libraries are distributed with the oneAPI Base Toolkit.
include_directories(${ONEAPI_ROOT}/dev-utilities/latest/include)
include_directories(SYSTEM "${SYCL_COMPILER_DIR}/../include/sycl" "${SYCL_COMPILER_DIR}/../include")

add_executable(rkRayTracerGPU src/rkRayTracerGPU.cpp)
if(MSVC)
# Set MSVS debugger environment variables so it is easier to attach the MSVS debugger after altering the application
    set_target_properties(rkRayTracerGPU
	    PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${embree_DIR}/../../../bin"
    )
endif(MSVC)
target_link_libraries(rkRayTracerGPU PRIVATE embree)
set_property(TARGET rkRayTracerGPU APPEND PROPERTY COMPILE_FLAGS "${SYCL_COMPILE_FLAGS} ${MSVC_OPT}")
set_property(TARGET rkRayTracerGPU APPEND PROPERTY LINK_FLAGS    "${SYCL_LINK_FLAGS}")

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
endif()

install(TARGETS rkRayTracerGPU
  DESTINATION ${CMAKE_INSTALL_BINDIR}
)
