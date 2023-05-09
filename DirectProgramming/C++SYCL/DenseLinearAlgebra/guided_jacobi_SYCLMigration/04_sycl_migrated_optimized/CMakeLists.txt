set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fsycl")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

add_executable (04_sycl_migrated_optimized src/main.cpp src/jacobi.cpp)
target_link_libraries(04_sycl_migrated_optimized sycl)

add_custom_target (run_smo ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/04_sycl_migrated_optimized)

