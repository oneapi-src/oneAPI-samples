set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

add_executable(jacobi_gpu_iterative_solver jacobi_gpu_iterative_solver.cpp)
target_link_libraries(jacobi_gpu_iterative_solver OpenCL sycl)

add_custom_target(run_gpu ./jacobi_gpu_iterative_solver)
