set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsycl --std=c++17")
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
add_executable(4_GPU_optimized 4_GPU_optimized.cpp)

target_link_libraries(1_CPU_only OpenCL sycl)
target_link_libraries(2_GPU_basic OpenCL sycl)
target_link_libraries(3_GPU_linear OpenCL sycl)
target_link_libraries(4_GPU_optimized OpenCL sycl)

add_custom_target(run_all 1_CPU_only 1024 1024 1024 100
                    COMMAND 2_GPU_basic 1024 1024 1024 100
                    COMMAND 3_GPU_linear 1024 1024 1024 100                    
                    COMMAND 4_GPU_optimized 1024 1024 1024 32 4 8 100)
add_custom_target(run_cpu 1_CPU_only 1024 1024 1024 100)
add_custom_target(run_gpu_basic 2_GPU_basic 1024 1024 1024 100)
add_custom_target(run_gpu_linear 3_GPU_linear 1024 1024 1024 100)
add_custom_target(run_gpu_optimized 4_GPU_optimized 1024 1024 1024 32 4 8 100)