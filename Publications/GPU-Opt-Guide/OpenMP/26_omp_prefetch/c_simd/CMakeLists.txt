add_compile_options(-fopenmp-target-simd)
add_link_options(-fopenmp-target-simd)
add_openmp_example(nbody_c_simd)
