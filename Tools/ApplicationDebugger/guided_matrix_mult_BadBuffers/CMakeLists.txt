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

set (a1_matrix_mul_zero_buff "a1_matrix_mul_zero_buff")
set (a2_matrix_mul "a2_matrix_mul")
set (b1_matrix_mul_null_usm "b1_matrix_mul_null_usm")
set (b2_matrix_mul_usm "b2_matrix_mul_usm")

add_executable (a1_matrix_mul_zero_buff src/a1_matrix_mul_zero_buff.cpp)
add_executable (a2_matrix_mul src/a2_matrix_mul.cpp)
add_executable (b1_matrix_mul_null_usm src/b1_matrix_mul_null_usm.cpp)
add_executable (b2_matrix_mul_usm src/b2_matrix_mul_usm.cpp)

target_link_libraries(a1_matrix_mul_zero_buff sycl)
target_link_libraries(a2_matrix_mul sycl)
target_link_libraries(b1_matrix_mul_null_usm sycl)
target_link_libraries(b2_matrix_mul_usm sycl)

add_custom_target(run_all a1_matrix_mul_zero_buff
    COMMAND a2_matrix_mul
	  COMMAND b1_matrix_mul_null_usm
    COMMAND b2_matrix_mul_usm)
add_custom_target(run_a1_zero_buff a1_matrix_mul_zero_buff)
add_custom_target(run_a2 a2_matrix_mul)
add_custom_target(run_b1_null_usm b1_matrix_mul_null_usm)
add_custom_target(run_b2_usm b2_matrix_mul_usm)
