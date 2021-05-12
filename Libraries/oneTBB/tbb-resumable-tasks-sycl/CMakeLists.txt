cmake_minimum_required (VERSION 3.0)

if (${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
    set(CMAKE_CXX_COMPILER "dpcpp-cl")
    set(CMAKE_C_COMPILER "dpcpp-cl")
else()
    set(CMAKE_CXX_COMPILER "dpcpp")
endif()

project (TBB-RESUMABLE-TASKS-SYCL)
add_subdirectory (src)
