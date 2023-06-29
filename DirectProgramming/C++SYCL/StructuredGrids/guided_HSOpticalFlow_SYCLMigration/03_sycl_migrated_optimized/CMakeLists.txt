set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
include_directories(${CMAKE_SOURCE_DIR}/03_sycl_migrated_optimized/Common/)
include_directories(${CMAKE_SOURCE_DIR}/03_sycl_migrated_optimized/include/)

add_executable (03_sycl_migrated_optimized Samples/5_Domain_Specific/HSOpticalFlow/main.cpp.dp.cpp Samples/5_Domain_Specific/HSOpticalFlow/flowSYCL.dp.cpp Samples/5_Domain_Specific/HSOpticalFlow/flowGold.cpp.dp.cpp)
target_link_libraries(03_sycl_migrated_optimized sycl)

add_custom_target (run_smo cd ${CMAKE_SOURCE_DIR}/03_sycl_migrated_optimized/ && ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/03_sycl_migrated_optimized)