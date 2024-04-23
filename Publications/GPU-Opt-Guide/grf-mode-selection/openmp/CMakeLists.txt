add_compile_options(-O3 -fiopenmp -fopenmp-targets=spir64 -mllvm -vpo-paropt-config=${CMAKE_CURRENT_SOURCE_DIR}/omp_example.yaml)
add_link_options(-fiopenmp -fopenmp-targets=spir64)

# add_example(grf-mode-selection-omp)
