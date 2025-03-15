add_compile_options(-O3 -fiopenmp -fopenmp-targets=spir64)
add_link_options(-fiopenmp -fopenmp-targets=spir64)
add_example(flat_openmp_01)
