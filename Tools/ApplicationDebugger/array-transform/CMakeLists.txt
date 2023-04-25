cmake_minimum_required (VERSION 3.4)
set (CMAKE_CXX_COMPILER "icpx")
project (array-transform LANGUAGES CXX)

if (NOT CMAKE_CXX_STANDARD)
  set (CMAKE_CXX_STANDARD 17)
endif ()

# This project requires the "Debug" build type.
if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
  set (CMAKE_BUILD_TYPE Debug)
  message (STATUS "Build Type is changed to 'Debug'.")
endif ()

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsycl")

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
message (STATUS "Optimization level is set to -O0.")

# Check that debugger executables exist in PATH
find_program(GDB gdb-oneapi)
if (NOT GDB)
  message (FATAL_ERROR "'gdb-oneapi' not found!")
endif ()

find_program(GDBSERVER gdbserver-ze)
if (NOT GDBSERVER)
  message (FATAL_ERROR "'gdbserver-ze' not found!")
endif ()

set(SYCL_COMPILE_TARGET "jit" CACHE STRING "Kernel compilation mode: {jit, cpu, fpga-emu, xe, ...}")
if (SYCL_COMPILE_TARGET MATCHES jit)
  # Do nothing.  We are already set.
elseif (SYCL_COMPILE_TARGET MATCHES cpu)
  set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsycl-targets=spir64_x86_64")
elseif (SYCL_COMPILE_TARGET MATCHES fpga-emu)
  set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsycl-targets=spir64_fpga")
else ()
  set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsycl-targets=spir64_gen")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Xs \"-device ${SYCL_COMPILE_TARGET} -internal_options -cl-kernel-debug-enable -options -cl-opt-disable\"")
endif ()
message (STATUS "SYCL_COMPILE_TARGET is '${SYCL_COMPILE_TARGET}'")

add_executable (${PROJECT_NAME} src/array-transform.cpp)

# Add custom target for starting a debug session
add_custom_target (debug-session
  ${GDB} ./${PROJECT_NAME}
  DEPENDS ${PROJECT_NAME})

# Add custom target for running
add_custom_target(run ONEAPI_DEVICE_SELECTOR=*:cpu ./${PROJECT_NAME}
  DEPENDS ${PROJECT_NAME})

add_custom_target(run-cpu ONEAPI_DEVICE_SELECTOR=*:cpu ./${PROJECT_NAME}
  DEPENDS ${PROJECT_NAME})

add_custom_target(run-gpu ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./${PROJECT_NAME}
  DEPENDS ${PROJECT_NAME})

add_custom_target(run-fpga ONEAPI_DEVICE_SELECTOR=*:fpga ./${PROJECT_NAME}
  DEPENDS ${PROJECT_NAME})
