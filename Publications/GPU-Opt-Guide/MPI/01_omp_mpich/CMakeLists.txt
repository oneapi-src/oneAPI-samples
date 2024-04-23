add_mpi_example(omp_mpich)
target_compile_options(omp_mpich PRIVATE -fiopenmp)
target_link_options(omp_mpich PRIVATE -fiopenmp)
