set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
include_directories(${CMAKE_SOURCE_DIR}/03_sycl_migrated/Common/)

add_executable (03_sycl_migrated src/main.cpp src/flowGold.cpp src/flowSYCL.cpp)
target_link_libraries(03_sycl_migrated sycl)

add_custom_target (run cd ${CMAKE_SOURCE_DIR}/03_sycl_migrated/ && ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/03_sycl_migrated)