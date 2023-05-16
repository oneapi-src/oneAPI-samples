set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
include_directories(${CMAKE_SOURCE_DIR}/02_sycl_dpct_migrated/Common/)

add_executable (02_sycl_dpct_migrated src/main.cpp src/oddEvenMergeSort.cpp src/sortingNetworks_validate.cpp)
target_link_libraries(02_sycl_dpct_migrated sycl)

add_custom_target (run_sdm  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/02_sycl_dpct_migrated)
