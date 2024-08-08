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
include_directories(${CMAKE_SOURCE_DIR}/02_sycl_migrated/Common/)
include_directories(${CMAKE_SOURCE_DIR}/02_sycl_migrated/include/)
add_executable (02_sycl_migrated Samples/0_Introduction/simpleAtomicIntrinsics/simpleAtomicIntrinsics_cpu.cpp Samples/0_Introduction/simpleAtomicIntrinsics/simpleAtomicIntrinsics.dp.cpp Samples/0_Introduction/simpleAtomicIntrinsics/simpleAtomicIntrinsics_kernel.dp.hpp)
target_link_libraries(02_sycl_migrated sycl)

add_custom_target (run_sm_gpu_cuda  cd ${CMAKE_SOURCE_DIR}/02_sycl_migrated/ && ONEAPI_DEVICE_SELECTOR=cuda:gpu ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/02_sycl_migrated)
add_custom_target (run_sm_cpu  cd ${CMAKE_SOURCE_DIR}/02_sycl_migrated/ && ONEAPI_DEVICE_SELECTOR=opencl:cpu ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/02_sycl_migrated)
add_custom_target (run_sm_gpu  cd ${CMAKE_SOURCE_DIR}/02_sycl_migrated/ && ONEAPI_DEVICE_SELECTOR=level_zero:gpu ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/02_sycl_migrated)
add_custom_target (run_sm_gpu_opencl  cd ${CMAKE_SOURCE_DIR}/02_sycl_migrated/ && ONEAPI_DEVICE_SELECTOR=opencl:gpu ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/02_sycl_migrated)

