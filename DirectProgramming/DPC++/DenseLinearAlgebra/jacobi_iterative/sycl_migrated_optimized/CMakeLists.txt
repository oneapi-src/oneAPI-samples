set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

add_executable (sycl_migrated_optimized src/main.cpp src/jacobi.cpp)
target_link_libraries(sycl_migrated_optimized sycl)

add_custom_target (run_smo_cpu ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/sycl_migrated_optimized)
add_custom_target (run_smo_gpu env SYCL_DEVICE_FILTER=gpu ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/sycl_migrated_optimized)

