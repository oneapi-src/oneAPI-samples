cmake_minimum_required(VERSION 3.12)

project(ParSTLTests)

find_package(TBB REQUIRED)
set(CMAKE_CXX_COMPILER icpx)

# Ensure either GPU or CPU is selected, default to CPU
#if(USE_GPU AND USE_CPU)
#   message(FATAL_ERROR "Cannot specify both USE_GPU and USE_CPU. Please choose one.")
#elseif(NOT USE_GPU AND NOT USE_CPU)
#   set(USE_CPU ON)  # Default to CPU if neither is set
#endif()

# Add executable
#add_executable(ParSTLTest main.cpp)

# Set compile options based on selected build target
#if(USE_GPU)
#   message(STATUS "Building for GPU")
#   target_compile_options(ParSTLTest PRIVATE -fsycl -fsycl-pstl-offload=gpu -D GPU)
#elseif(USE_CPU)
#   message(STATUS "Building for CPU")
#   target_compile_options(ParSTLTest PRIVATE -fsycl -D CPU)
#endif()

# Link SYCL and TBB libraries
#target_link_libraries(ParSTLTest PRIVATE sycl tbb)
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
