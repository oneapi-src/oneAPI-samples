cmake_minimum_required (VERSION 3.0)
set (CMAKE_CXX_COMPILER "dpcpp")
project (array-transform LANGUAGES CXX)

if (NOT CMAKE_CXX_STANDARD)
  set (CMAKE_CXX_STANDARD 17)
endif ()

# This project requires the "Debug" build type.
if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
  set (CMAKE_BUILD_TYPE Debug)
  message (WARNING "Build Type is changed to 'Debug'.")
endif ()

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
message (WARNING "Optimization level is set to -O0.")

# Check that debugger executables exist in PATH
find_program(GDB gdb-oneapi)
if (NOT GDB)
  message (FATAL_ERROR "'gdb-oneapi' not found!")
endif ()

find_program(GDBSERVER gdbserver-gt)
if (NOT GDBSERVER)
  message (FATAL_ERROR "'gdbserver-gt' not found!")
endif ()

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
