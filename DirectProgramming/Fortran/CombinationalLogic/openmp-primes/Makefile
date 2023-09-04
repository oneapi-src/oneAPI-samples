## =============================================================
## Copyright © 2020 Intel Corporation
##
## SPDX-License-Identifier: MIT
## =============================================================
##
##
##******************************************************************************
## Content:
##
##  Build for openmp_sample
##******************************************************************************
SHELL := /bin/bash

FC = ifx

release: openmp_sample

debug: openmp_sample_dbg

run: release ; time ./openmp_sample

debug_run: debug ; time ./openmp_sample_dbg

openmp_sample: openmp_sample.o
	$(FC) -O2 -qopenmp $^ -o $@

openmp_sample_dbg: openmp_sample_dbg.o
	$(FC) -O0 -g -qopenmp $^ -o $@

%.o: src/%.f90
	$(FC) -O2 -c -qopenmp -o $@  $<

%_dbg.o: src/%.f90
	$(FC) -O0 -g -c -qopenmp -o $@  $<

clean:
	/bin/rm -f core.* *.o openmp_sample openmp_sample_dbg
