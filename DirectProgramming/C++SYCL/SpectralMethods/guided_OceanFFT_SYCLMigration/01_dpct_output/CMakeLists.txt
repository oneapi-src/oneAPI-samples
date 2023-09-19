if(INTEL_MAX_GPU)
  #To build for Intel® Data Center GPU Max 1550 or 1100
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=intel_gpu_pvc -D INTEL_MAX_GPU")
  set(CMAKE_EXE_LINKER_FLAGS "-qmkl ${CMAKE_EXE_LINKER_FLAGS}")
elseif (NVIDIA_GPU)
  #To build for Nvidia GPU like P100, A100, H100 ...
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -D NVIDIA_GPU")
  set(CMAKE_EXE_LINKER_FLAGS "-L $ENV{ONEMKL_INSTALL_DIR}/lib/ -lonemkl ${CMAKE_EXE_LINKER_FLAGS}")
  include_directories($ENV{ONEMKL_INSTALL_DIR}/include/)
else()
  #To build for Intel® UHD Graphics, Intel® Gen9, Gen11, Xeon CPU 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl ")
set(CMAKE_EXE_LINKER_FLAGS "-qmkl ${CMAKE_EXE_LINKER_FLAGS}")
endif()

include_directories(${CMAKE_SOURCE_DIR}/01_dpct_output/Common/)
include_directories(${CMAKE_SOURCE_DIR}/01_dpct_output/include/)

add_executable (01_dpct_output Samples/4_CUDA_Libraries/oceanFFT/oceanFFT_kernel.dp.cpp Samples/4_CUDA_Libraries/oceanFFT/oceanFFT.cpp.dp.cpp)
target_link_libraries(01_dpct_output sycl)


add_custom_target (run cd ${CMAKE_SOURCE_DIR}/01_dpct_output/ && ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/01_dpct_output qatest)
