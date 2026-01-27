set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -DWITH_FUSED_BETA=1 ")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
include_directories(${CMAKE_SOURCE_DIR}/01_dpct_output/include/)

add_executable (01_dpct_output longstaff_schwartz_svd_2.dp.cpp)
target_link_libraries(01_dpct_output sycl)

add_custom_target (run cd ${CMAKE_SOURCE_DIR}/01_dpct_output/ && ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/01_dpct_output)
