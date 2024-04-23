## =============================================================
## Copyright © 2024 Intel Corporation
##
## SPDX-License-Identifier: MIT
## =============================================================
##
##
##******************************************************************************
## Content:
##
##  Build for vec_sample
##****************************************************************************** 
#
FC=ifx
FFLAGS=-O2 -qopt-report=2
OBJ=src/driver.o src/matvec.o

all : matvec

run : matvec
	./matvec

src/%.o: src/%.f90
	$(FC) $(FFLAGS) -c $< -o $@

matvec: $(OBJ)
	$(FC) -V $^ -o matvec

clean:
	-rm -f matvec $(OBJ) src/*.optrpt src/*.opt.yaml
