##=============================================================
## Copyright © 2020 Intel Corporation
##
## SPDX-License-Identifier: MIT
## =============================================================
##
##
##******************************************************************************
## To compile and run the sequential version type: make seq
## To compile and run the CPU parallel version type: make par
## To compile and run the GPU version type: make gpu
##******************************************************************************

FC=ifx
FCFLAGS_CPU=-xhost
FCFLAGS_PAR=-xhost -qopenmp
FCFLAGS_GPU=-xhost -qopenmp -fopenmp-targets=spir64

SEQ=./seq
SEQ_OBJS= 01_mm_CPU_sequential.o
PAR=./par
PAR_OBJS= 02_mm_CPU_parallel.o
GPU=./gpu
GPU_OBJS= 03_mm_GPU.o

$(SEQ): $(SEQ_OBJS)
	$(FC) $(FCFLAGS_CPU) $(SEQ_OBJS) -o $@
	$(SEQ)

$(PAR): $(PAR_OBJS)
	$(FC) $(FCFLAGS_PAR) $(PAR_OBJS) -o $@
	$(PAR)

$(GPU): $(GPU_OBJS)
	$(FC) $(FCFLAGS_GPU) $(GPU_OBJS) -o $@
	$(GPU)

all: $(SEQ) $(PAR) $(GPU)

01_mm_CPU_sequential.o: src/01_mm_CPU_sequential.f90
	$(FC) $(FCFLAGS_CPU) -c $< -o $@

02_mm_CPU_parallel.o: src/02_mm_CPU_parallel.f90
	$(FC) $(FCFLAGS_PAR) -c $< -o $@

03_mm_GPU.o: src/03_mm_GPU.f90
	$(FC) $(FCFLAGS_GPU) -c $< -o $@

.DEFAULT_GOAL=$(SEQ)

.PHONY: clean

clean:
	-/bin/rm -f *.o $(SEQ) $(PAR) $(GPU)
