if(INTEL_MAX_GPU)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=intel_gpu_pvc -D INTEL_MAX_GPU")
elseif(NVIDIA_GPU)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -D NVIDIA_GPU")
else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl ")
endif()
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
include_directories(${CMAKE_SOURCE_DIR}/02_sycl_migrated_optimized/Common/)
include_directories(${CMAKE_SOURCE_DIR}/02_sycl_migrated_optimized/include/)

add_executable(02_sycl_migrated_optimized Samples/5_Domain_Specific/quasirandomGenerator/quasirandomGenerator.cpp.dp.cpp Samples/5_Domain_Specific/quasirandomGenerator/quasirandomGenerator_gold.cpp Samples/5_Domain_Specific/quasirandomGenerator/quasirandomGenerator_kernel.dp.cpp)
target_link_libraries(02_sycl_migrated_optimized sycl)

add_custom_target (run cd ${CMAKE_SOURCE_DIR}/02_sycl_migrated_optimized/ && ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/02_sycl_migrated_optimized)

