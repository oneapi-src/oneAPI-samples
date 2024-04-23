##=============================================================
## Copyright © 2020 Intel Corporation
##
## SPDX-License-Identifier: MIT
## =============================================================
##
##**************************************************************
## To compile and run the do concurrent examples: make run_dc
## To compile and run the for-loop examples: make run_omp
## To compile and run all examples: make run_all
##**************************************************************

default: run_all

run_all: run_dc run_omp

run_dc: img_seg_do_conc_cpu_seq img_seg_do_conc_cpu_par img_seg_do_conc_gpu
	./img_seg_do_conc_cpu_seq -n 12 -o 2 -i 1 -d
	./img_seg_do_conc_cpu_par -n 12 -o 2 -i 1 -d
	OMP_TARGET_OFFLOAD=MANDATORY ./img_seg_do_conc_gpu -n 12 -o 2 -i 1 -d

run_omp: img_seg_cpu img_seg_omp_cpu img_seg_omp_gpu
	./img_seg_cpu -n 12 -o 2 -i 1 -d
	./img_seg_omp_cpu -n 12 -o 2 -i 1 -d
	OMP_TARGET_OFFLOAD=MANDATORY ./img_seg_omp_gpu -n 12 -o 2 -i 1 -d

OMP_OPTS = -qopenmp
GPU_OPTS = -fopenmp-targets=spir64 -fopenmp-target-do-concurrent

img_seg_do_conc_cpu_seq: img_seg_do_concurrent.F90
	ifx $< -o $@

img_seg_do_conc_cpu_par: img_seg_do_concurrent.F90
	ifx $< -o $@ $(OMP_OPTS)

img_seg_do_conc_gpu: img_seg_do_concurrent.F90
	ifx $< -o $@ $(OMP_OPTS) $(GPU_OPTS)

img_seg_cpu: img_seg_omp_target.F90
	ifx $< -o $@

img_seg_omp_cpu: img_seg_omp_target.F90
	ifx $< -o $@ $(OMP_OPTS)

img_seg_omp_gpu: img_seg_omp_target.F90
	ifx $< -o $@ $(OMP_OPTS) $(GPU_OPTS)

clean:
	-rm -f img_seg_do_conc_cpu_seq img_seg_do_conc_cpu_par img_seg_do_conc_gpu
	-rm -f img_seg_cpu img_seg_omp_cpu img_seg_omp_gpu

.PHONY: clean all run_all run_dc run_omp
