set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
include_directories(${CMAKE_SOURCE_DIR}/dpct_sycl_migrated/Common/)

add_executable (dpct_sycl_migrated src/main.cpp src/jacobi.cpp)
target_link_libraries(dpct_sycl_migrated sycl)

add_custom_target (run_dsm_cpu ./dpct_sycl_migrated)
add_custom_target (run_dsm_gpu SYCL_DEVICE_FILTER=gpu ./dpct_sycl_migrated)
