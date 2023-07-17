set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
include_directories(${CMAKE_SOURCE_DIR}/02_sycl_migrated/Common/)
include_directories(${CMAKE_SOURCE_DIR}/02_sycl_migrated/include/)

add_executable (02_sycl_migrated Samples/5_Domain_Specific/HSOpticalFlow/main.cpp.dp.cpp Samples/5_Domain_Specific/HSOpticalFlow/flowSYCL.dp.cpp Samples/5_Domain_Specific/HSOpticalFlow/flowGold.cpp.dp.cpp)
target_link_libraries(02_sycl_migrated sycl)

add_custom_target (run cd ${CMAKE_SOURCE_DIR}/02_sycl_migrated/ && ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/02_sycl_migrated)