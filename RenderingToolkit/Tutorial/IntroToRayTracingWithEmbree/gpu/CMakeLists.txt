cmake_minimum_required(VERSION 3.16)
project(rkRayTracerGPU LANGUAGES CXX)

set(ONEAPI_ROOT "")
if(DEFINED ENV{ONEAPI_ROOT})
  set(ONEAPI_ROOT "$ENV{ONEAPI_ROOT}")
  message(STATUS "ONEAPI_ROOT FROM ENVIRONMENT: ${ONEAPI_ROOT}")
else()
  message(FATAL_ERROR "ONEAPI_ROOT not set. Please use a vars script (.bat/.sh) from the oneAPI deployment directory")
endif(DEFINED ENV{ONEAPI_ROOT})

set(EMBREE_BASE_DIR "")
set(RKCOMMON_BASE_DIR "")
set(TBB_BASE_DIR "")
set(COMPILERRUNTIMES_BASE_DIR "")
set(PATHADDITIONS_DIRS "")
set(STB_HEADERS_DIR ${CMAKE_SOURCE_DIR})
if(EXISTS ${ONEAPI_ROOT}/oneapi-vars.sh OR EXISTS ${ONEAPI_ROOT}/oneapi-vars.bat)
  set(EMBREE_BASE_DIR ${ONEAPI_ROOT})
  set(RKCOMMON_BASE_DIR ${ONEAPI_ROOT})
  #no MacOS
  set(TBB_BASE_DIR ${ONEAPI_ROOT})
  #Compiler runtimes help for windows MSVS debugger DLL help only
  set(COMPILERRUNTIMES_BASE_DIR ${ONEAPI_ROOT})
  set(PATHADDITIONS_DIRS "PATH=%PATH%;${ONEAPI_ROOT}/bin")
else()
  set(EMBREE_BASE_DIR ${ONEAPI_ROOT}/embree/latest)
  set(RKCOMMON_BASE_DIR ${ONEAPI_ROOT}/rkcommon/latest)
  #no MacOS
  set(TBB_BASE_DIR ${ONEAPI_ROOT}/tbb/latest)
  #Compiler runtimes help for windows MSVS debugger DLL help only
  set(COMPILERRUNTIMES_BASE_DIR ${ONEAPI_ROOT}/compiler/latest)
endif(EXISTS ${ONEAPI_ROOT}/oneapi-vars.sh OR EXISTS ${ONEAPI_ROOT}/oneapi-vars.bat)

find_package(embree 4.0 REQUIRED PATHS ${ONEAPI_ROOT})

find_package(IntelSYCL REQUIRED)

if(PATHADDITIONS_DIRS STREQUAL "")
  set(PATHADDITIONS_DIRS "PATH=%PATH%;${COMPILERRUNTIMES_BASE_DIR}/bin;${embree_DIR}/../../../bin;")
endif(PATHADDITIONS_DIRS STREQUAL "")

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


#stb headers are located in dev-utilities with the old oneAPI dirs layout
include_directories(${STB_HEADERS_DIR})
include_directories(SYSTEM "${SYCL_COMPILER_DIR}/../include/sycl" "${SYCL_COMPILER_DIR}/../include")

add_executable(rkRayTracerGPU src/rkRayTracerGPU.cpp)
if(MSVC)
# Set MSVS debugger environment variables so it is easier to attach the MSVS debugger after altering the application
    set_target_properties(rkRayTracerGPU
	    PROPERTIES VS_DEBUGGER_ENVIRONMENT "${PATHADDITIONS_DIRS}"
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
