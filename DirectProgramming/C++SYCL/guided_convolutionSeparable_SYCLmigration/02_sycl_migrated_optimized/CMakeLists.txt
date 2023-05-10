set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
include_directories(${CMAKE_SOURCE_DIR}/02_sycl_migrated_optimized/Common/)
include_directories(${CMAKE_SOURCE_DIR}/02_sycl_migrated_optimized/include/)

add_executable (02_sycl_migrated_optimized Samples/2_Concepts_and_Techniques/convolutionSeparable/main.cpp.dp.cpp Samples/2_Concepts_and_Techniques/convolutionSeparable/convolutionSeparable.dp.cpp Samples/2_Concepts_and_Techniques/convolutionSeparable/convolutionSeparable_gold.cpp)
target_link_libraries(02_sycl_migrated_optimized sycl)

add_custom_target (run_smo cd ${CMAKE_SOURCE_DIR}/02_sycl_migrated_optimized/ && ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/02_sycl_migrated_optimized)
