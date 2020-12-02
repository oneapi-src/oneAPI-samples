cmake_minimum_required (VERSION 3.5)
project (GEMM)

set(CMAKE_CXX_COMPILER dpcpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mkl")

add_executable(gemm src/gemm.cpp)

add_custom_target(run ./gemm)
