cmake_minimum_required (VERSION 3.4)
set (CMAKE_CXX_COMPILER "icpx")
project (jacobi LANGUAGES CXX)

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

set (PROJECT_NAME_BUGGED "${PROJECT_NAME}-bugged")
set (PROJECT_NAME_FIXED "${PROJECT_NAME}-fixed")

add_executable (${PROJECT_NAME_BUGGED} src/jacobi.cpp)
add_executable (${PROJECT_NAME_FIXED} src/jacobi.cpp)
target_compile_definitions (${PROJECT_NAME_FIXED} PRIVATE FIXED)

# Add custom target for starting a debug session
add_custom_target (debug-session
  ${GDB} ./${PROJECT_NAME_BUGGED}
  DEPENDS ${PROJECT_NAME_BUGGED})

# Add custom target for running
add_custom_target(run ONEAPI_DEVICE_SELECTOR=*:cpu ./${PROJECT_NAME_BUGGED}
  DEPENDS ${PROJECT_NAME_BUGGED})

add_custom_target(run-cpu-fixed ONEAPI_DEVICE_SELECTOR=*:cpu ./${PROJECT_NAME_FIXED}
  DEPENDS ${PROJECT_NAME_FIXED})
add_custom_target(run-cpu ONEAPI_DEVICE_SELECTOR=*:cpu ./${PROJECT_NAME_BUGGED}
  DEPENDS ${PROJECT_NAME_BUGGED})

add_custom_target(run-gpu-fixed ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./${PROJECT_NAME_FIXED}
  DEPENDS ${PROJECT_NAME_FIXED})
add_custom_target(run-gpu ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./${PROJECT_NAME_BUGGED}
  DEPENDS ${PROJECT_NAME_BUGGED})

add_custom_target(run-fpga-fixed ONEAPI_DEVICE_SELECTOR=*:fpga ./${PROJECT_NAME_FIXED}
  DEPENDS ${PROJECT_NAME_FIXED})
add_custom_target(run-fpga ONEAPI_DEVICE_SELECTOR=*:fpga ./${PROJECT_NAME_BUGGED}
  DEPENDS ${PROJECT_NAME_BUGGED})
