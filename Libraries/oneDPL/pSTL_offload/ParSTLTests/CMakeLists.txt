cmake_minimum_required(VERSION 3.12)

project(ParSTLTests)

find_package(TBB REQUIRED)
set(CMAKE_CXX_COMPILER icpx)

if(GPU)
  #To build for Intel® Data Center GPU Max 1550 or 1100
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-pstl-offload=gpu -D GPU")
endif()
if(CPU)
  #To build for Intel® UHD Graphics, Intel® Gen9, Gen11, Xeon CPU 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D CPU ")
endif()
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "-ltbb ${CMAKE_EXE_LINKER_FLAGS}")

add_executable (ParSTLTest main.cpp)
