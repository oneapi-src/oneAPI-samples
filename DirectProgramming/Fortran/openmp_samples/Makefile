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

FC = ifort

release: openmp_sample.exe

debug: openmp_sample_dbg.exe

run: release ; @export  DYLD_LIBRARY_PATH="$(LIBRARY_PATH)" ; ./openmp_sample.exe

debug_run: debug ; @export  DYLD_LIBRARY_PATH="$(LIBRARY_PATH)" ; ./openmp_sample_dbg.exe

openmp_sample.exe: openmp_sample.o
	$(FC) -O2 -fpp -qopenmp $^ -o $@

openmp_sample_dbg.exe: openmp_sample_dbg.o
	$(FC) -O0 -g -fpp -qopenmp $^ -o $@

%.o: src/%.f90
	$(FC) -O2 -c -fpp -qopenmp -o $@  $<

%_dbg.o: src/%.f90
	$(FC) -O0 -g -c -fpp -qopenmp -o $@  $<

clean:
	/bin/rm -f core.* *.o *.exe
