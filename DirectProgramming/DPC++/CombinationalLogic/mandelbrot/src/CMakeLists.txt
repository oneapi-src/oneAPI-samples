set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

add_executable(mandelbrot main.cpp)
target_link_libraries(mandelbrot OpenCL sycl)
add_custom_target(run ./mandelbrot)

add_executable(mandelbrot_usm main.cpp)
target_compile_definitions(mandelbrot_usm PRIVATE MANDELBROT_USM)
target_link_libraries(mandelbrot_usm OpenCL sycl)
add_custom_target(run_usm ./mandelbrot_usm)
