cmake_minimum_required (VERSION 3.4)
set (CMAKE_CXX_COMPILER "icpx")
project (matrix_mul LANGUAGES CXX)

if (NOT CMAKE_CXX_STANDARD)
  set (CMAKE_CXX_STANDARD 17)
endif ()

# This project requires the "Debug" build type.
if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
  set (CMAKE_BUILD_TYPE Debug)
  message (STATUS "Build Type is changed to 'Debug'.")
endif ()

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsycl -O0")
message (STATUS "Optimization level is set to -O0.")

# Check that debugger executables exist in PATH
find_program(GDB gdb-oneapi)
if (NOT GDB)
  message (FATAL_ERROR "'gdb-oneapi' not found!")
endif ()

set (1_matrix_mul_SLM_size "1_matrix_mul_SLM_size")
set (2_matrix_mul "$2_matrix_mul")

add_executable (1_matrix_mul_SLM_size src/1_matrix_mul_SLM_size.cpp)
add_executable (2_matrix_mul src/2_matrix_mul.cpp)

# Add custom target for starting a debug session
add_custom_target (debug-session
  ${GDB} ./${1_matrix_mul_SLM_size}
  DEPENDS ${1_PROJECT_NAMl_SLM_size})

target_link_libraries(1_matrix_mul_SLM_size sycl)
target_link_libraries(2_matrix_mul sycl)

add_custom_target(run_bugged 1_matrix_mul_SLM_size)
add_custom_target(run_OK 2_matrix_mul)
add_custom_target(run_all 1_matrix_mul_SLM_size
	COMMAND 2_matrix_mul)

