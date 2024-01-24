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
include_directories(${CMAKE_SOURCE_DIR}/02_sycl_migrated_option1/Common/)
include_directories(${CMAKE_SOURCE_DIR}/02_sycl_migrated_option1/include/)

add_executable (02_sycl_migrated_option1 Samples/3_CUDA_Features/simpleCudaGraphs/simpleCudaGraphs.dp.cpp)
target_link_libraries(02_sycl_migrated_option1 sycl)

add_custom_target (run_op1 cd ${CMAKE_SOURCE_DIR}/02_sycl_migrated_option1/ && ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/02_sycl_migrated_option1)