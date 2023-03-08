set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "-qmkl ${CMAKE_EXE_LINKER_FLAGS}")
include_directories(${CMAKE_SOURCE_DIR}/04_sycl_migrated_optimized/Common/)

add_executable (04_sycl_migrated_optimized src/MonteCarlo_gold.cpp src/MonteCarlo_kernel.cpp src/MonteCarloMultiGPU.cpp src/multithreading.cpp)
target_link_libraries(04_sycl_migrated_optimized sycl)

add_custom_target (run_smo_cpu cd ${CMAKE_SOURCE_DIR}/04_sycl_migrated_optimized/ && ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/04_sycl_migrated_optimized)
add_custom_target (run_smo_gpu SYCL_DEVICE_FILTER=gpu cd ${CMAKE_SOURCE_DIR}/04_sycl_migrated_optimized/ && ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/04_sycl_migrated_optimized)
