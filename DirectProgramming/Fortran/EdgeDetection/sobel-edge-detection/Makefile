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

run_dc: sobel_do_conc_cpu_seq sobel_do_conc_cpu_par sobel_do_conc_gpu
	./sobel_do_conc_cpu_seq -i xeon_4gen.ppm -o edges_dc_seq.ppm
	./sobel_do_conc_cpu_par -i xeon_4gen.ppm -o edges_dc_par.ppm
	OMP_TARGET_OFFLOAD=MANDATORY ./sobel_do_conc_gpu -i xeon_4gen.ppm -o edges_dc_gpu.ppm

run_omp: sobel_seq sobel_omp_cpu sobel_omp_gpu
	./sobel_seq -i xeon_4gen.ppm -o edges_seq.ppm
	./sobel_omp_cpu -i xeon_4gen.ppm -o edges_omp_par.ppm
	OMP_TARGET_OFFLOAD=MANDATORY ./sobel_omp_gpu -i xeon_4gen.ppm -o edges_omp_gpu.ppm

OMP_OPTS = -qopenmp
GPU_OPTS = -fopenmp-targets=spir64 -fopenmp-target-do-concurrent

sobel_do_conc_cpu_seq: sobel_do_concurrent.F90
	ifx ppm_image_io.F90 $< -o $@

sobel_do_conc_cpu_par: sobel_do_concurrent.F90
	ifx ppm_image_io.F90 $< -o $@ $(OMP_OPTS)

sobel_do_conc_gpu: sobel_do_concurrent.F90
	ifx ppm_image_io.F90 $< -o $@ $(OMP_OPTS) $(GPU_OPTS)

sobel_seq: sobel_omp_target.F90
	ifx ppm_image_io.F90 $< -o $@

sobel_omp_cpu: sobel_omp_target.F90
	ifx ppm_image_io.F90 $< -o $@ $(OMP_OPTS)

sobel_omp_gpu: sobel_omp_target.F90
	ifx ppm_image_io.F90 $< -o $@ $(OMP_OPTS) $(GPU_OPTS)

clean:
	-rm -f sobel_do_conc_cpu_seq sobel_do_conc_cpu_par sobel_do_conc_gpu
	-rm -f sobel_seq sobel_omp_cpu sobel_omp_gpu
	-rm -f edges*.ppm *.mod *.modmic

.PHONY: clean all run_all run_dc run_omp
