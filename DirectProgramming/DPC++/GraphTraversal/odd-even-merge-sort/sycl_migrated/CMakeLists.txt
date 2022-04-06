set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
include_directories(${CMAKE_SOURCE_DIR}/sycl_migrated/common/)

add_executable (sycl_migrated src/main.cpp src/odd_even_merge_sort.cpp src/sorting_networks_validate.cpp)
target_link_libraries(sycl_migrated sycl)

add_custom_target (run_cpu ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/sycl_migrated)
add_custom_target (run_gpu SYCL_DEVICE_FILTER=gpu ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/sycl_migrated)
