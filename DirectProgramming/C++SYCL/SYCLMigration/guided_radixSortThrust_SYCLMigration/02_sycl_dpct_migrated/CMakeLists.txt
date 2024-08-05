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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -std=c++17")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core")

include_directories(${CMAKE_SOURCE_DIR}/02_sycl_dpct_migrated/Common/)
include_directories(${CMAKE_SOURCE_DIR}/02_sycl_dpct_migrated/include/)
add_subdirectory("src")
