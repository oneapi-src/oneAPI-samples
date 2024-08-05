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

FC = ifx -O0
#FC = ifx -O1
#FC = ifx -O2
#FC = ifx -O3

OBJS = int_sin.o

all: int_sin

run: int_sin
	./int_sin

int_sin: $(OBJS)
	ifx $^ -o $@

%.o: src/%.f90
	$(FC) $^ -c

clean:
	/bin/rm -f $(OBJS) int_sin
