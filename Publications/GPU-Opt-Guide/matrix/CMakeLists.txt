set(_sources matrix.cpp multiply.cpp)
add_executable(matrix ${_sources})
add_sycl_to_target(TARGET matrix SOURCES ${_sources})
