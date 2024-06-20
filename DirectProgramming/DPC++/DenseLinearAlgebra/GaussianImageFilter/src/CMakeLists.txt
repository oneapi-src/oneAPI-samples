set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fsycl")
set(CMAKE_BUILD_TYPE "RelWithDebInfo")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
add_executable (gaussian_blur gaussian_blur.cpp)
target_link_libraries(gaussian_blur OpenCL sycl)

