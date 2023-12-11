if(NVIDIA_GPU)
  #To build for Nvidia GPU like P100, A100, H100 ...
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -D NVIDIA_GPU")
else()
  #To build for Intel® UHD Graphics, Intel® Gen9, Gen11, Xeon CPU 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl ")
endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
include_directories(${CMAKE_SOURCE_DIR}/02_sycl_migrated_optimized/Common/)
include_directories(${CMAKE_SOURCE_DIR}/02_sycl_migrated_optimized/include/)

add_executable (02_sycl_migrated_optimized Samples/5_Domain_Specific/HSOpticalFlow/main.cpp.dp.cpp Samples/5_Domain_Specific/HSOpticalFlow/flowSYCL.dp.cpp Samples/5_Domain_Specific/HSOpticalFlow/flowGold.cpp.dp.cpp)
target_link_libraries(02_sycl_migrated_optimized sycl)

add_custom_target (run cd ${CMAKE_SOURCE_DIR}/02_sycl_migrated_optimized/ && ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/02_sycl_migrated_optimized )