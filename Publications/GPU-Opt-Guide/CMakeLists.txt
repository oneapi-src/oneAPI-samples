cmake_minimum_required(VERSION 3.21)
set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_Fortran_COMPILER ifx)

project(GPUOptGuide)

include(CheckLanguage)
enable_testing()

find_package(IntelDPCPP REQUIRED)

check_language(Fortran)
if(CMAKE_Fortran_COMPILER)
  enable_language(Fortran)
else()
  message(STATUS "No Fortran support. Disabling Fortran tests. Install oneAPI HPC Toolkit.")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/")
include(SetLibraryPath)

message(STATUS ${MKL_INCLUDE_DIR})
message(STATUS ${MKL_LIBRARY_DIR})
message(STATUS ${TBB_INCLUDE_DIR})
message(STATUS ${TBB_LIBRARY_DIR})

find_library(mkl_core NAMES mkl_core PATHS ${MKL_LIBRARY_DIR} REQUIRED)
find_library(mkl_sycl NAMES mkl_sycl PATHS ${MKL_LIBRARY_DIR} REQUIRED)
find_library(mkl_intel_ilp64 NAMES mkl_intel_ilp64 PATHS ${MKL_LIBRARY_DIR} REQUIRED)
find_library(mkl_tbb_thread NAMES mkl_tbb_thread PATHS ${MKL_LIBRARY_DIR} REQUIRED)
find_library(tbb NAMES tbb PATHS ${TBB_LIBRARY_DIR} REQUIRED)

string(CONCAT WARNING_CXX_FLAGS_STR
  "-Wall "
  "-Wextra "
  "-Winit-self "
  "-Wunused-function "
  "-Wuninitialized "
  "-Wmissing-declarations "
  "-fdiagnostics-color=auto "
)
string(REPLACE " " ";" COMMON_CXX_FLAGS "${WARNING_CXX_FLAGS_STR}")

function(add_example_with_mkl name)
  add_executable(${name} ${name}.cpp)
  target_compile_options(${name} PRIVATE ${COMMON_CXX_FLAGS})
  target_compile_options(${name} PRIVATE -fiopenmp -fopenmp-targets=spir64 -qmkl=parallel)
  target_link_directories(${name}
    PUBLIC ${MKL_LIBRARY_DIR} ${TBB_LIBRARY_DIR}
  )
  target_link_libraries(${name}
    PRIVATE ${mkl_sycl}
  )
  target_link_options(${name} PRIVATE -fiopenmp -fopenmp-targets=spir64 -qmkl=parallel -fsycl -lmkl_sycl -fsycl-device-code-split=per_kernel)
  add_test(NAME ${name} COMMAND ${name} ${ARGN})
endfunction(add_example_with_mkl)

function(add_fortran_example_with_mkl name)
  if(CMAKE_Fortran_COMPILER)
    add_executable(${name} ${name}.f)
    target_compile_options(${name} PRIVATE -warn all)
    target_compile_options(${name} PRIVATE -fiopenmp -fopenmp-targets=spir64 -qmkl=parallel -fpp -free)
    target_link_directories(${name}
      PUBLIC ${MKL_LIBRARY_DIR}
    )
    target_link_libraries(${name}
      PRIVATE ${mkl_sycl}
    )
    target_link_options(${name} PRIVATE -fiopenmp -fopenmp-targets=spir64 -qmkl=parallel -fsycl -lmkl_sycl)
    add_test(NAME ${name} COMMAND ${name} ${ARGN})
  endif()
endfunction(add_fortran_example_with_mkl)

function(add_example name)
  add_executable(${name} ${name}.cpp)
  target_compile_options(${name} PRIVATE ${COMMON_CXX_FLAGS})
  target_link_options(${name} PRIVATE -fsycl-device-code-split=per_kernel)
  add_test(NAME ${name} COMMAND ${name} ${ARGN})
endfunction(add_example)

function(add_fortran_example name)
  if(CMAKE_Fortran_COMPILER)
    add_executable(${name} ${name}.f90)
    target_compile_options(${name} PRIVATE -warn all)
    target_compile_options(${name} PRIVATE -fiopenmp -fopenmp-targets=spir64)
    target_link_options(${name} PRIVATE -fiopenmp -fopenmp-targets=spir64)
    add_test(NAME ${name} COMMAND ${name} ${ARGN})
  endif()
endfunction(add_fortran_example)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(matrix)
add_subdirectory(exec-model)
add_subdirectory(io-kernel)
add_subdirectory(jitting)
add_subdirectory(kernels)
add_subdirectory(memory-movement)
add_subdirectory(restrict)
add_subdirectory(slm)
add_subdirectory(usm)
add_subdirectory(sub-group)
add_subdirectory(buffers)
add_subdirectory(buffer-accessors)
add_subdirectory(reduction)
add_subdirectory(conditionals)
add_subdirectory(libraries-kernel)
add_subdirectory(libraries-stdlib)
add_subdirectory(libraries-fcorr)
add_subdirectory(multiple-queue-submission)
add_subdirectory(multiple-devices)
add_subdirectory(multiple-kernel-execution)
add_subdirectory(work-group-size)
add_subdirectory(registers)
add_subdirectory(OpenMP)
