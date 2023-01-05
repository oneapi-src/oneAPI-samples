set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fsycl")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
add_executable (sepia sepia_sycl.cpp)
target_link_libraries(sepia OpenCL sycl)
file(COPY ../input/silverfalls1.png DESTINATION .)
if(WIN32)
add_custom_target (run sepia.exe silverfalls1.png)
else()
add_custom_target (run ${CMAKE_COMMAND} -E env SYCL_BE=PI_OPENCL ./sepia silverfalls1.png)
endif()


