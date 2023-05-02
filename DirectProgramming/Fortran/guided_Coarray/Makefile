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
##  Build for guided_coarray
##
##  To compile and run the sequential version type: make seq
##  To compile and run your coarray version type: make coarray
##  To compile and run supplied coarray version type: make final
##******************************************************************************

FC=ifx
FCFLAGS=
FCFLAGS_COARRAY=-coarray

SEQ=seq
SEQ_OBJS= mcpi_sequential.o
COARRAY=coarray
COARRAY_OBJS= mcpi_coarray.o
FINAL=final
COARRAY_FINAL_OBJS= mcpi_coarray_final.o

mcpi_sequential.o: mcpi_sequential.F90
mcpi_coarray.o: mcpi_coarray.F90
mcpi_coarray_final.o: mcpi_coarray_final.F90

$(SEQ): $(SEQ_OBJS)
	$(FC) $(SEQ_OBJS) -o $@
	$(SEQ)

$(COARRAY): $(COARRAY_OBJS)
	$(FC) $(FCFLAGS_COARRAY) $(COARRAY_OBJS) -o $@
	$(COARRAY)

$(FINAL): $(COARRAY_FINAL_OBJS)
	$(FC) $(FCFLAGS_COARRAY) $(COARRAY_FINAL_OBJS) -o $@
	$(FINAL)

all: $(SEQ) $(COARRAY) $(FINAL)

mcpi_sequential.o: mcpi_sequential.F90
	$(FC) $(FCFLAGS) -c $< -o $@

mcpi_coarray.o: mcpi_coarray.F90
	$(FC) $(FCFLAGS_COARRAY) -c $< -o $@

mcpi_coarray_final.o: mcpi_coarray_final.F90
	$(FC) $(FCFLAGS_COARRAY) -c $< -o $@

.DEFAULT_GOAL=$(SEQ)

.PHONY: clean

clean:
	-/bin/rm -f *.o $(SEQ) $(COARRAY) $(FINAL)
