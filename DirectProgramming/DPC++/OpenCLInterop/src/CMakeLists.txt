set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

file(COPY ../src/vector_add_kernel.cl DESTINATION .)
add_executable(prog dpcpp_with_opencl_objects.dp.cpp)
target_link_libraries(prog OpenCL sycl)
add_custom_target(run ./prog)