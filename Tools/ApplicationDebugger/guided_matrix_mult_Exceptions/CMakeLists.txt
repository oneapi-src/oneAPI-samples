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

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsycl -g -O0")
message (STATUS "Optimization level is set to -O0.")

# Check that debugger executables exist in PATH
find_program(GDB gdb-oneapi)
if (NOT GDB)
  message (FATAL_ERROR "'gdb-oneapi' not found!")
endif ()

set (1_matrix_mul_null_pointer "1_matrix_mul_null_pointer")
set (2_matrix_mul_multi_offload "2_matrix_mul_multi_offload")
set (3_matrix_mul "3_matrix_mul")

add_executable (1_matrix_mul_null_pointer src/1_matrix_mul_null_pointer.cpp)
add_executable (2_matrix_mul_multi_offload src/2_matrix_mul_multi_offload.cpp)
add_executable (3_matrix_mul src/3_matrix_mul.cpp)


target_link_libraries(1_matrix_mul_null_pointer sycl)
target_link_libraries(2_matrix_mul_multi_offload sycl)
target_link_libraries(3_matrix_mul sycl)


add_custom_target(run_all 1_matrix_mul_null_pointer
        COMMAND 2_matrix_mul_multi_offload
        COMMAND 3_matrix_mul)
add_custom_target(run_1_null_pointer 1_matrix_mul_null_pointer)
add_custom_target(run_2_multi_offload 2_matrix_mul_multi_offload)
add_custom_target(run_3 3_matrix_mul)
