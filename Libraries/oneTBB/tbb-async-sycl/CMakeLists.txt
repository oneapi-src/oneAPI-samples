cmake_minimum_required (VERSION 3.4)
project (TBB-ASYNC-SYCL LANGUAGES CXX)

if (${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
    set(CMAKE_CXX_COMPILER "dpcpp-cl")
else()
    set(CMAKE_CXX_COMPILER "icpx")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
endif()

add_subdirectory (src)
