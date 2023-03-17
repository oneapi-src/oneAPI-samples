cmake_minimum_required(VERSION 3.16)
project(rkPathTracer LANGUAGES CXX)

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

#tbb is used for tasking in this example
find_package(TBB REQUIRED PATHS ${ONEAPI_ROOT})

find_package(embree 4.0 REQUIRED PATHS ${ONEAPI_ROOT}/embree/latest/lib/cmake NO_DEFAULT_PATH)

#rkcommon objects for rendering are used in this example
find_package(rkcommon REQUIRED CONFIG PATHS ${ONEAPI_ROOT}/rkcommon/latest/lib/cmake NO_DEFAULT_PATH)

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

#stb headers are located in dev-utilities. oneAPI 2022.2 release and earlier: these header only libraries are distributed with the oneAPI Base Toolkit.
include_directories(${ONEAPI_ROOT}/dev-utilities/latest/include)

set(HEADERS src/CornellBox.h  src/DefaultCubeAndPlane.h  src/Geometry.h  src/Lights.h src/Materials.h src/PathTracer.h src/Pool.h src/Renderer.h src/SceneGraph.h src/Sphere.h src/definitions.h src/RandomSampler.h)
add_executable(rkPathTracer src/rkPathTracer.cpp ${HEADERS})

if(MSVC)
# Set MSVS debugger environment variables so it is easier to attach the MSVS debugger after altering the application
    set_target_properties(rkPathTracer
	    PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${embree_DIR}/../../../bin;${rkcommon_DIR}/../../../bin;${tbb_DIR}/../../../bin;"
    )
endif(MSVC)
target_link_libraries(rkPathTracer PRIVATE embree TBB::tbb rkcommon::rkcommon)

install(TARGETS rkPathTracer
  DESTINATION ${CMAKE_INSTALL_BINDIR}
)
