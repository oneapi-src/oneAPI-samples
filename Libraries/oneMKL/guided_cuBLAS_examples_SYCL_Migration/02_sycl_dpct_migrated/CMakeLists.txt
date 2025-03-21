set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -std=c++17 -qmkl")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -qmkl -qmkl-sycl-impl=blas")

include_directories(${CMAKE_SOURCE_DIR}/02_sycl_dpct_migrated/Common/)
add_subdirectory("Level-1")
add_subdirectory("Level-2")
add_subdirectory("Level-3")
