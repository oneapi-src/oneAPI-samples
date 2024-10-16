set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -Xarch_device -fsanitize=address -g -O2")
# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
set(ENV{ONEAPI_DEVICE_SELECTOR} "level_zero:cpu")
# set(ENV{ONEAPI_DEVICE_SELECTOR} "level_zero:gpu")

add_executable(array_reduction array_reduction.cpp)
add_executable(bad_free bad_free.cpp)
add_executable(device_global device_global.cpp)
add_executable(group_local group_local.cpp)
add_executable(local_stencil local_stencil.cpp)
add_executable(map map.cpp)
add_executable(matmul_broadcast matmul_broadcast.cpp)
add_executable(misalign-long misalign-long.cpp)
add_executable(nd_range_reduction nd_range_reduction.cpp)

target_link_libraries(array_reduction OpenCL sycl)
target_link_libraries(bad_free OpenCL sycl)
target_link_libraries(device_global OpenCL sycl)
target_link_libraries(group_local OpenCL sycl)
target_link_libraries(local_stencil OpenCL sycl)
target_link_libraries(map OpenCL sycl)
target_link_libraries(matmul_broadcast OpenCL sycl)
target_link_libraries(misalign-long OpenCL sycl)
target_link_libraries(nd_range_reduction OpenCL sycl)

add_custom_target(run_all array_reduction
                    COMMAND bad_free
                    COMMAND device_global
                    COMMAND group_local
                    COMMAND local_stencil
                    COMMAND map
                    COMMAND matmul_broadcast
                    COMMAND misalign-long
                    COMMAND nd_range_reduction)

add_custom_target(run_array_reduction array_reduction)
add_custom_target(run_bad_free bad_free)
add_custom_target(run_device_global device_global)
add_custom_target(run_group_local group_local)
add_custom_target(run_local_stencil local_stencil)
add_custom_target(run_map map)
add_custom_target(run_matmul_broadcast matmul_broadcast)
add_custom_target(run_misalign-long misalign-long)
add_custom_target(run_nd_range_reduction nd_range_reduction)
