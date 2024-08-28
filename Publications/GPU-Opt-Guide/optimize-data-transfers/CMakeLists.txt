add_compile_options(-O3 -fiopenmp -fopenmp-targets=spir64)
add_link_options(-fiopenmp -fopenmp-targets=spir64)

add_example(sycl_prepare_bench)
add_fortran_example_with_mkl(openmp_system_mem)
add_fortran_example_with_mkl(openmp_host_usm)
add_fortran_example_with_mkl(openmp_register_mem)
