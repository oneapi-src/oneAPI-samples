# Makefile for Linux

default: run_all

all: run_all

run_all: lu_solve lu_solve_omp_offload lu_solve_omp_offload_optimized
	./lu_solve -n 64 -b 8 -r 1 -c 2
	./lu_solve_omp_offload -n 64 -b 8 -r 1 -c 2
	./lu_solve_omp_offload_optimized -n 64 -b 8 -r 1 -c 2

IFX_OPTS = -i8 -free -qmkl
IFX_OPTS_OFFLOAD = -DMKL_ILP64 -qopenmp -fopenmp-targets=spir64 -fsycl -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl

lu_solve: lu_solve_omp_offload.F90
	ifx $< -o $@ $(IFX_OPTS)

lu_solve_omp_offload: lu_solve_omp_offload.F90
	ifx $< -o $@ $(IFX_OPTS) $(IFX_OPTS_OFFLOAD)

lu_solve_omp_offload_optimized: lu_solve_omp_offload_optimized.F90
	ifx $< -o $@ $(IFX_OPTS) $(IFX_OPTS_OFFLOAD)

clean:
	-rm -f lu_solve lu_solve_omp_offload lu_solve_omp_offload_optimized *.mod *.modmic

.PHONY: run_all clean all
