cmake_minimum_required (VERSION 3.4)
set (CMAKE_CXX_COMPILER "dpcpp")
project (array-transform LANGUAGES CXX)

if (NOT CMAKE_CXX_STANDARD)
  set (CMAKE_CXX_STANDARD 17)
endif ()

# This project requires the "Debug" build type.
if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
  set (CMAKE_BUILD_TYPE Debug)
  message (STATUS "Build Type is changed to 'Debug'.")
endif ()

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
message (STATUS "Optimization level is set to -O0.")

# Check that debugger executables exist in PATH
find_program(GDB gdb-oneapi)
if (NOT GDB)
  message (FATAL_ERROR "'gdb-oneapi' not found!")
endif ()

find_program(GDBSERVER gdbserver-gt)
if (NOT GDBSERVER)
  message (FATAL_ERROR "'gdbserver-gt' not found!")
endif ()

set(DPCPP_COMPILE_TARGET "jit" CACHE STRING "Kernel compilation mode: {jit, cpu, fpga-emu, kbl, gen9, gen12LP, ...}")
if (DPCPP_COMPILE_TARGET MATCHES jit)
  # Do nothing.  We are already set.
elseif (DPCPP_COMPILE_TARGET MATCHES cpu)
  set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsycl-targets=spir64_x86_64-unknown-linux-sycldevice")
elseif (DPCPP_COMPILE_TARGET MATCHES fpga-emu)
  set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsycl-targets=spir64_fpga-unknown-linux-sycldevice")
else ()
  set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsycl-targets=spir64_gen-unknown-linux-sycldevice")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Xs \"-device ${DPCPP_COMPILE_TARGET} -internal_options -cl-kernel-debug-enable -options -cl-opt-disable\"")
endif ()
message (STATUS "DPCPP_COMPILE_TARGET is '${DPCPP_COMPILE_TARGET}'")

add_executable (${PROJECT_NAME} src/array-transform.cpp)

# Add custom target for starting a debug session
add_custom_target (debug-session
  ${GDB} ./${PROJECT_NAME}
  DEPENDS ${PROJECT_NAME})

# Add custom target for running
add_custom_target(run ./${PROJECT_NAME} cpu
  DEPENDS ${PROJECT_NAME})

add_custom_target(run-cpu ./${PROJECT_NAME} cpu
  DEPENDS ${PROJECT_NAME})

add_custom_target(run-gpu ./${PROJECT_NAME} gpu
  DEPENDS ${PROJECT_NAME})

add_custom_target(run-fpga ./${PROJECT_NAME} accelerator
  DEPENDS ${PROJECT_NAME})
