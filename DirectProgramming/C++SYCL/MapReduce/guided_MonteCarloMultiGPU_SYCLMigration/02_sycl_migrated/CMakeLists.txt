set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "-pthread ${CMAKE_EXE_LINKER_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "-qmkl ${CMAKE_EXE_LINKER_FLAGS}")
include_directories(${CMAKE_SOURCE_DIR}/02_sycl_migrated/Common/)
include_directories(${CMAKE_SOURCE_DIR}/02_sycl_migrated/include/)

add_executable (02_sycl_migrated Samples/5_Domain_Specific/MonteCarloMultiGPU/MonteCarloMultiGPU.cpp.dp.cpp Samples/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_gold.cpp.dp.cpp Samples/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_kernel.dp.cpp Samples/5_Domain_Specific/MonteCarloMultiGPU/multithreading.cpp)
target_link_libraries(02_sycl_migrated sycl)

add_custom_target (run_sm_cpu  cd ${CMAKE_SOURCE_DIR}/02_sycl_migrated/ && ONEAPI_DEVICE_SELECTOR=opencl:cpu ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/02_sycl_migrated)
add_custom_target (run_sm_gpu  cd ${CMAKE_SOURCE_DIR}/02_sycl_migrated/ && ONEAPI_DEVICE_SELECTOR=level_zero:gpu ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/02_sycl_migrated)
add_custom_target (run_sm_gpu_opencl  cd ${CMAKE_SOURCE_DIR}/02_sycl_migrated/ && ONEAPI_DEVICE_SELECTOR=opencl:gpu ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/02_sycl_migrated)

