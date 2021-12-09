project(ispc_gsg)
cmake_minimum_required(VERSION 3.1)

set(ONEAPI_ROOT "")
if(WIN32)
    add_compile_definitions(WIN32)
    if(MSVC)
        add_compile_options(/EHsc)
    endif(MSVC)
endif(WIN32)

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

if (NOT DEFINED ISPC_EXECUTABLE)
    find_program (ISPC_EXECUTABLE ispc)
    if (NOT ISPC_EXECUTABLE)
        message(FATAL_ERROR "Failed to find ispc" )
    endif()
endif(NOT DEFINED ISPC_EXECUTABLE)
message(STATUS "ISPC_EXECUTABLE: ${ISPC_EXECUTABLE}")

set (ISPC_SRC_NAME "simple")
set (TARGET_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${ISPC_SRC_NAME}.cpp)
set (ISPC_FLAGS -O2)
set (ISPC_IA_TARGETS "sse2-i32x4")
set (ISPC_ARM_TARGETS "neon")
set (ISPC_OBJ_NAME "${CMAKE_CURRENT_BINARY_DIR}/${ISPC_SRC_NAME}_ispc${CMAKE_CXX_OUTPUT_EXTENSION}")
set (ISPC_HEADER_NAME "${CMAKE_CURRENT_BINARY_DIR}/include-single/${ISPC_SRC_NAME}_ispc.h")
set (OUTPUT_TARGET "${ISPC_IA_TARGETS}")


list(APPEND ISPC_BUILD_OUTPUT ${ISPC_HEADER_NAME} ${ISPC_OBJ_NAME} )
# ISPC command
add_custom_command(OUTPUT ${ISPC_BUILD_OUTPUT}
    COMMAND ${ISPC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/simple.ispc ${ISPC_FLAGS} --target=${ISPC_IA_TARGETS} --arch=x86_64
                                   -h ${ISPC_HEADER_NAME} -o ${ISPC_OBJ_NAME}
    VERBATIM
    DEPENDS ${ISPC_EXECUTABLE}
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/${ISPC_SRC_NAME}.ispc")


add_executable(simple ${ISPC_BUILD_OUTPUT} "${CMAKE_CURRENT_SOURCE_DIR}/src/${ISPC_SRC_NAME}.ispc")
set_target_properties(simple PROPERTIES LINKER_LANGUAGE CXX)
target_sources(simple PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/simple.cpp)
#ISPC emits a header for the ISPC module. The header location is included below
target_include_directories(simple PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include-single)

#Set variables for multiple isa extension targeting.
#Targets with mask and gang
set (ISPC_IA_TARGETS_MG "sse2-i32x4,sse4-i32x4,avx1-i32x8,avx2-i32x8,avx512skx-i32x8")
set (ISPC_IA_TARGETS sse2 sse4 avx avx2 avx512skx)
set (ISPC_ARM_TARGETS "neon")
set (ISPC_OBJ_NAME "${CMAKE_CURRENT_BINARY_DIR}/${ISPC_SRC_NAME}_ispc_multi${CMAKE_CXX_OUTPUT_EXTENSION}")
set (ISPC_HEADER_NAME "${CMAKE_CURRENT_BINARY_DIR}/include-multi/${ISPC_SRC_NAME}_ispc.h")
set (OUTPUT_TARGET "${ISPC_IA_TARGETS}")

list(APPEND ISPC_MULTI_BUILD_OUTPUT ${ISPC_HEADER_NAME} ${ISPC_OBJ_NAME} )
foreach(TARGET_TOKEN IN LISTS ISPC_IA_TARGETS)
  list(APPEND ISPC_MULTI_BUILD_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${ISPC_SRC_NAME}_ispc_multi_${TARGET_TOKEN}${CMAKE_CXX_OUTPUT_EXTENSION}")
endforeach()

add_custom_command(OUTPUT ${ISPC_MULTI_BUILD_OUTPUT}
    COMMAND ${ISPC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/${ISPC_SRC_NAME}.ispc ${ISPC_FLAGS} --target=${ISPC_IA_TARGETS_MG} --arch=x86_64
                                   -h ${ISPC_HEADER_NAME} -o ${ISPC_OBJ_NAME}
    VERBATIM
    DEPENDS ${ISPC_EXECUTABLE}
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/${ISPC_SRC_NAME}.ispc")

add_executable(simple_multi ${ISPC_MULTI_BUILD_OUTPUT} "${CMAKE_CURRENT_SOURCE_DIR}/src/${ISPC_SRC_NAME}.ispc")
set_target_properties(simple_multi PROPERTIES LINKER_LANGUAGE CXX)
target_sources(simple_multi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/simple.cpp)
#ISPC emits a header for the ISPC module. The header location is included below
target_include_directories(simple_multi PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include-multi)
