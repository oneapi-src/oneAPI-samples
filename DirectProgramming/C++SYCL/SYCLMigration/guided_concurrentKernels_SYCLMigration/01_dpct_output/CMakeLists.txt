set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
include_directories(${CMAKE_SOURCE_DIR}/01_dpct_output/Common/)

add_executable (01_dpct_output Samples/0_Introduction/concurrentKernels/concurrentKernels.dp.cpp)
target_link_libraries(01_dpct_output sycl)

add_custom_target (run cd ${CMAKE_SOURCE_DIR}/01_dpct_output/ && ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/01_dpct_output)
