set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

add_executable(prog1 dpcpp_with_opencl_kernel.dp.cpp)
target_link_libraries(prog1 OpenCL sycl)
add_custom_target(run_prog1 ./prog1)

file(COPY ../src/vector_add_kernel.cl DESTINATION .)
add_executable(prog2 dpcpp_with_opencl_objects.dp.cpp)
target_link_libraries(prog2 OpenCL sycl)
add_custom_target(run_prog2 ./prog2)