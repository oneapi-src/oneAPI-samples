cmake_minimum_required(VERSION 3.16)
project(rkPathTracer LANGUAGES CXX)

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
if(APPLE)
  set(TBB_BASE_DIR ${ONEAPI_ROOT}/etc/rkcommon/tbb)
else()
  set(TBB_BASE_DIR ${ONEAPI_ROOT})
endif(APPLE)
  #Compiler runtimes help for windows MSVS debugger DLL help only
  set(COMPILERRUNTIMES_BASE_DIR ${ONEAPI_ROOT})
  set(PATHADDITIONS_DIRS "PATH=%PATH%;${ONEAPI_ROOT}/bin")
else()
  set(EMBREE_BASE_DIR ${ONEAPI_ROOT}/embree/latest)
  set(RKCOMMON_BASE_DIR ${ONEAPI_ROOT}/rkcommon/latest)
if(APPLE)
  set(TBB_BASE_DIR ${ONEAPI_ROOT}/rkcommon/latest/etc/rkcommon/tbb)
else()
  set(TBB_BASE_DIR ${ONEAPI_ROOT}/tbb/latest)
endif(APPLE)
  #Compiler runtimes help for windows MSVS debugger DLL help only
  set(COMPILERRUNTIMES_BASE_DIR ${ONEAPI_ROOT}/compiler/latest)
endif(EXISTS ${ONEAPI_ROOT}/oneapi-vars.sh OR EXISTS ${ONEAPI_ROOT}/oneapi-vars.bat)

#tbb is used for tasking in this example
find_package(TBB REQUIRED PATHS ${TBB_BASE_DIR}/lib/cmake NO_DEFAULT_PATH)

find_package(embree 4.0 REQUIRED PATHS ${EMBREE_BASE_DIR}/lib/cmake NO_DEFAULT_PATH)

#rkcommon objects for rendering are used in this example
find_package(rkcommon REQUIRED CONFIG PATHS ${RKCOMMON_BASE_DIR}/lib/cmake NO_DEFAULT_PATH)

if(PATHADDITIONS_DIRS STREQUAL "")
  set(PATHADDITIONS_DIRS "PATH=%PATH%;${COMPILERRUNTIMES_BASE_DIR}/bin;${embree_DIR}/../../../bin;${rkcommon_DIR}/../../../bin;${tbb_DIR}/../../../bin;")
endif(PATHADDITIONS_DIRS STREQUAL "")


if(MSVC)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
else()
  set(CMAKE_CXX_STANDARD 14)
endif(MSVC)

# Set default build type to RelWithDebInfo if not specified
if(NOT MSVC)
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set. Using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)
endif(NOT MSVC)

#stb headers are located in dev-utilities with the old oneAPI dires layout
include_directories(${STB_HEADERS_DIR})

set(HEADERS src/CornellBox.h  src/DefaultCubeAndPlane.h  src/Geometry.h  src/Lights.h src/Materials.h src/PathTracer.h src/Pool.h src/Renderer.h src/SceneGraph.h src/Sphere.h src/definitions.h src/RandomSampler.h)
add_executable(rkPathTracer src/rkPathTracer.cpp ${HEADERS})

if(MSVC)
  message(STATUS "Using ${PATHADDITIONS_DIRS} for the MSVS Debugger env")
# Set MSVS debugger environment variables so it is easier to attach the MSVS debugger after altering the application
  set_target_properties(rkPathTracer
    PROPERTIES VS_DEBUGGER_ENVIRONMENT ${PATHADDITIONS_DIRS}
  )
endif(MSVC)
target_link_libraries(rkPathTracer PRIVATE embree TBB::tbb rkcommon::rkcommon)

install(TARGETS rkPathTracer
  DESTINATION ${CMAKE_INSTALL_BINDIR}
)
