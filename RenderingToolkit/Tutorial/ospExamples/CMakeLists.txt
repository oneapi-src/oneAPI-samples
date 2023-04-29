## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(OSPRAY_EXAMPLES LANGUAGES CXX)

include(CMakePrintHelpers)
include(GNUInstallDirs)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH
  ${PROJECT_SOURCE_DIR}/cmake/dependencies
)


set(INSTALL_DIR_ABSOLUTE "Release")
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

list(APPEND CMAKE_PREFIX_PATH "${ONEAPI_ROOT}/rkcommon/latest")
find_package(ospray REQUIRED PATHS ${ONEAPI_ROOT})

if(MSVC)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif(MSVC)

include_directories(${OSPRAY_INCLUDE_DIR} ${RKCOMMON_INCLUDE_DIRS})
link_directories(${OSPRAY_ROOT}/lib ${ONEAPI_ROOT}/rkcommon/latest/lib)

include(cmake/ospray_macros.cmake)

include(glfw)


set(OpenGL_GL_PREFERENCE "LEGACY")
find_package(OpenGL 2 REQUIRED)

link_libraries(ospray rkcommon)

# prefer libGL over libOpenGl for better compatibility with SWR
set(OpenGL_GL_PREFERENCE "LEGACY")
add_subdirectory(arcball_camera)
add_subdirectory(external)

add_executable(ospExamples
  ${OSPRAY_RESOURCE}
  GLFWOSPRayWindow.cpp
  imgui_impl_glfw_gl3.cpp
  ospExample.cpp

  ospray_testing/ospray_testing.h

  ospray_testing/ospray_testing.inl
  ospray_testing/builder.h
  ospray_testing/Noise.h
  ospray_testing/ospray_testing.cpp

  ospray_testing/Boxes.cpp
  ospray_testing/Interpolation.cpp
  ospray_testing/Builder.cpp
  ospray_testing/CornellBox.cpp
  ospray_testing/Curves.cpp
  ospray_testing/Empty.cpp
  ospray_testing/GravitySpheresVolume.cpp
  ospray_testing/Noise.cpp
  ospray_testing/PerlinNoiseVolumes.cpp
  ospray_testing/RandomSpheres.cpp
  ospray_testing/Streamlines.cpp
  ospray_testing/SubdivisionCube.cpp
  ospray_testing/UnstructuredVolumeSimple.cpp
  ospray_testing/UnstructuredVolumeGen.cpp
  ospray_testing/Planes.cpp
  ospray_testing/ClippingGeometries.cpp
  ospray_testing/ParticleVolume.cpp
  ospray_testing/VdbVolume.cpp
  ospray_testing/Transparency.cpp
  ospray_testing/Instancing.cpp

  # regression test scenes
  ospray_testing/test_pt_glass.cpp
  ospray_testing/test_pt_luminous.cpp
  ospray_testing/test_pt_metal_roughness.cpp
  ospray_testing/test_pt_metallic_flakes.cpp
  ospray_testing/test_pt_obj.cpp

  ospray_testing/rawToAMR.cpp
  ospray_testing/rawToAMR.h
)

target_link_libraries(ospExamples
PUBLIC arcball_camera
PUBLIC imgui
PUBLIC glfw
  ${OPENGL_LIBRARIES}
)

ospray_sign_target(ospExamples)

install(TARGETS ospExamples
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT apps
)
