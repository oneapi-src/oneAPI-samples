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
##  Build for optimize_sample
##******************************************************************************
#
# >>>>> SET OPTIMIZATION LEVEL BELOW <<<<<
#
#Uncomment one of the following with which you wish to compile

FC = ifort -O0 
#FC = ifort -O1 
#FC = ifort -O2 
#FC = ifort -O3 

OBJS = int_sin.o

all: int_sin

run: int_sin
	./int_sin

int_sin: $(OBJS)
	ifort $^ -o $@

%.o: src/%.f90
	$(FC) $^ -c

clean:
	/bin/rm -f core.* $(OBJS) int_sin

