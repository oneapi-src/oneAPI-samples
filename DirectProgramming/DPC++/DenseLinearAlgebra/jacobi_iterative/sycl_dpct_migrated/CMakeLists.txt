set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
include_directories(${CMAKE_SOURCE_DIR}/sycl_dpct_migrated/Common/)

add_executable (sycl_dpct_migrated src/main.cpp src/jacobi.cpp)
target_link_libraries(sycl_dpct_migrated sycl)

add_custom_target (run_sdm_cpu ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/sycl_dpct_migrated)
add_custom_target (run_sdm_gpu SYCL_DEVICE_FILTER=gpu ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/sycl_dpct_migrated)
