if(INTEL_MAX_GPU)
  #To build for Intel® Data Center GPU Max 1550 or 1100
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=intel_gpu_pvc -D INTEL_MAX_GPU")
elseif (NVIDIA_GPU)
  #To build for Nvidia GPU like P100, A100, H100 ...
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -D NVIDIA_GPU")
else()
  #To build for Intel® UHD Graphics, Intel® Gen9, Gen11, Xeon CPU 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl ")
endif()
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "-pthread ${CMAKE_EXE_LINKER_FLAGS}")
include_directories(${CMAKE_SOURCE_DIR}/03_sycl_migrated_optimized/Common/)
include_directories(${CMAKE_SOURCE_DIR}/03_sycl_migrated_optimized/include/)

add_executable (03_sycl_migrated_optimized Samples/3_CUDA_Features/jacobiCudaGraphs/jacobi.dp.cpp Samples/3_CUDA_Features/jacobiCudaGraphs/main.cpp.dp.cpp)
target_link_libraries(03_sycl_migrated_optimized sycl)

add_custom_target (run_smo0 cd ${CMAKE_SOURCE_DIR}/03_sycl_migrated_optimized/ && ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/03_sycl_migrated_optimized gpumethod=0)

add_custom_target (run_smo1 cd ${CMAKE_SOURCE_DIR}/03_sycl_migrated_optimized/ && ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/03_sycl_migrated_optimized gpumethod=1)