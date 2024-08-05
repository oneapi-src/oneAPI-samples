add_compile_options(-mcmodel=medium -O3 -fiopenmp -fopenmp-targets=spir64)
add_link_options(-mcmodel=medium -fiopenmp -fopenmp-targets=spir64)

add_subdirectory(small)
add_subdirectory(large)
