set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "-pthread ${CMAKE_EXE_LINKER_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "-qmkl ${CMAKE_EXE_LINKER_FLAGS}")
include_directories(${CMAKE_SOURCE_DIR}/01_dpct_output/Common/)
include_directories(${CMAKE_SOURCE_DIR}/01_dpct_output/include/)

add_executable (01_dpct_output Samples/5_Domain_Specific/MonteCarloMultiGPU/MonteCarloMultiGPU.cpp.dp.cpp Samples/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_gold.cpp.dp.cpp Samples/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_kernel.dp.cpp Samples/5_Domain_Specific/MonteCarloMultiGPU/multithreading.cpp)
target_link_libraries(01_dpct_output sycl)

add_custom_target (run_cpu cd ${CMAKE_SOURCE_DIR}/01_dpct_output/ && SYCL_DEVICE_FILTER=cpu ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/01_dpct_output)
add_custom_target (run_gpu cd ${CMAKE_SOURCE_DIR}/01_dpct_output/ && SYCL_DEVICE_FILTER=level_zero:gpu ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/01_dpct_output)
add_custom_target (run_gpu_opencl cd ${CMAKE_SOURCE_DIR}/01_dpct_output/ && SYCL_DEVICE_FILTER=opencl:gpu ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/01_dpct_output)
