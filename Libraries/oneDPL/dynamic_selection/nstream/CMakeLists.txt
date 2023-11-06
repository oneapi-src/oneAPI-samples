cmake_minimum_required (VERSION 3.4)

if(UNIX)
    # Direct CMake to use icpx rather than the default C++ compiler/linker
    set(CMAKE_CXX_COMPILER icpx)
else() # Windows
    # Force CMake to use icx-cl rather than the default C++ compiler/linker
    # (needed on Windows only)
    include (CMakeForceCompiler)
    CMAKE_FORCE_CXX_COMPILER (icx-cl IntelDPCPP)
    include (Platform/Windows-Clang)
endif()

project(nstreams_ds CXX)

set(CMAKE_CXX_FLAGS "-fsycl -Wall")

add_executable(1_nstreams_sycl ./src/1_nstreams_sycl.cpp)
add_executable(2_nstreams_policies ./src/2_nstreams_policies.cpp)

target_link_libraries(1_nstreams_sycl sycl)
target_link_libraries(2_nstreams_policies sycl)

add_custom_target(run_sycl 1_nstreams_sycl 1000)
add_custom_target(run_policies 2_nstreams_policies 1000 1)
add_custom_target(run_all 1_nstreams_sycl 1000
  COMMAND 2_nstreams_policies 1000 1)
