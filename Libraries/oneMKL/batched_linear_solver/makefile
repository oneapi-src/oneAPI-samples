# Makefile for Linux

default: run_all

all: run_all

run_all: lu_solve lu_solve_omp_offload lu_solve_omp_offload_optimized
	./lu_solve -n 64 -b 8 -r 1 -c 2
	./lu_solve_omp_offload -n 64 -b 8 -r 1 -c 2
	./lu_solve_omp_offload_optimized -n 64 -b 8 -r 1 -c 2

IFX_OPTS = -i8 -free -qmkl
IFX_OPTS_OFFLOAD = -DMKL_ILP64 -qopenmp -fopenmp-targets=spir64 -qmkl-sycl-impl=lapack

lu_solve: lu_solve_omp_offload.F90
	ifx $< -o $@ $(IFX_OPTS)

lu_solve_omp_offload: lu_solve_omp_offload.F90
	ifx $< -o $@ $(IFX_OPTS) $(IFX_OPTS_OFFLOAD)

lu_solve_omp_offload_optimized: lu_solve_omp_offload_optimized.F90
	ifx $< -o $@ $(IFX_OPTS) $(IFX_OPTS_OFFLOAD)

clean:
	-rm -f lu_solve lu_solve_omp_offload lu_solve_omp_offload_optimized *.mod *.modmic

.PHONY: run_all clean all
