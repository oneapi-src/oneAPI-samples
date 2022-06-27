set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 --std=c++17")
# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

add_executable(1_CPU_only 1_CPU_only.cpp)
add_executable(2_GPU_basic 2_GPU_basic.cpp)
add_executable(3_GPU_linear 3_GPU_linear.cpp)
add_executable(4_GPU_private_memory_I 4_GPU_private_memory_I.cpp)
add_executable(4_GPU_private_memory_K 4_GPU_private_memory_K.cpp)
add_executable(5_GPU_optimized 5_GPU_optimized.cpp)

target_link_libraries(1_CPU_only OpenCL sycl)
target_link_libraries(2_GPU_basic OpenCL sycl)
target_link_libraries(3_GPU_linear OpenCL sycl)
target_link_libraries(4_GPU_private_memory_I OpenCL sycl)
target_link_libraries(4_GPU_private_memory_K OpenCL sycl)
target_link_libraries(5_GPU_optimized OpenCL sycl)

add_custom_target(run_all 1_CPU_only 256 256 256 100
                    COMMAND 2_GPU_basic 256 256 256 100
                    COMMAND 3_GPU_linear 256 256 256 100
                    COMMAND 4_GPU_private_memory_I 256 256 256 100
                    COMMAND 5_GPU_optimized 256 256 256 100 256 16 16)
add_custom_target(run_cpu 1_CPU_only 256 256 256 100)
add_custom_target(run_gpu_basic 2_GPU_basic 256 256 256 100)
add_custom_target(run_gpu_linear 3_GPU_linear 256 256 256 100)
add_custom_target(run_gpu_private_I 4_GPU_private_memory_I 256 256 256 100)
add_custom_target(run_gpu_private_K 4_GPU_private_memory_K 256 256 256 100)
add_custom_target(run_gpu_optimized 5_GPU_optimized 256 256 256 100 256 16 16)
add_custom_target(run_gpu_optimized_good_params 5_GPU_optimized 256 256 256 100 16 8 16)
add_custom_target(run_gpu_optimized_poor_params 5_GPU_optimized 256 256 256 100 1 16 16)

add_custom_target(run_verify_all 1_CPU_only 256 256 256 100
                    COMMAND 2_GPU_basic 256 256 256 100 1
                    COMMAND 3_GPU_linear 256 256 256 100 1
                    COMMAND 4_GPU_private_memory_I 256 256 256 100 1
                    COMMAND 5_GPU_optimized 256 256 256 100 256 16 16 1)
add_custom_target(run_verify_cpu 1_CPU_only 256 256 256 100 1)
add_custom_target(run_verify_gpu_basic 2_GPU_basic 256 256 256 100 1)
add_custom_target(run_verify_gpu_linear 3_GPU_linear 256 256 256 100 1)
add_custom_target(run_verify_gpu_private 4_GPU_private_memory_I 256 256 256 100 1)
add_custom_target(run_verify_gpu_optimized 5_GPU_optimized 256 256 256 100 16 8 16 1)

