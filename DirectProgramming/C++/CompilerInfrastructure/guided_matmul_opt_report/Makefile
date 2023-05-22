#==============================================================
#
# SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
# http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
#
# Copyright (C) Intel Corporation
#
# THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
#
# =============================================================

CXX=icx
INCLUDES=-I src
CXXFLAGS=-c -O1 $(INCLUDES) $(BITS)

PROGS=MatVector

help:
	@echo "Usage:"
	@echo "    To build: make <config>"
	@echo "    To run  : MatVector                (for baseline)"
	@echo "              vec_samples_<config>.out (for other configs)"
	@echo ""
	@echo "where <config> is one of the following:"
	@echo "    baseline, vec_report1, vec_report2, pointer_disam, align, and ipo"
	@echo ""

run: help

baseline: clean Driver.o Multiply.o
	$(CXX) Driver.o Multiply.o -o MatVector $(LIBS)
	@echo "Build complete."
	@echo "You may run ./MatVector now."

vec_report1: CXXFLAGS = -c -O2 -std=c99 -qopt-report=1 -qopt-report-phase=vec -D NOFUNCCALL $(INCLUDES) $(BITS)
vec_report1: clean Driver.o Multiply.o
	$(CXX) Driver.o Multiply.o -o vec_samples_report1.out $(LIBS)
	@echo "Build complete."
	@echo "You may run ./vec_samples_report1.out now."
	@echo "LIBS is $(LIBS), BITS is $(BITS)."


vec_report2: CXXFLAGS = -c -O2 -std=c99 -qopt-report=2 -qopt-report-phase=vec -D NOFUNCCALL $(INCLUDES) $(BITS)
vec_report2: clean Driver.o Multiply.o
	$(CXX) Driver.o Multiply.o -o vec_samples_report2.out $(LIBS)
	@echo "Build complete."
	@echo "You may run ./vec_samples_report2.out now."

pointer_disam: CXXFLAGS = -c -O2 -std=c99 -qopt-report=2 -qopt-report-phase=vec -D NOALIAS $(INCLUDES) $(BITS)
pointer_disam: clean Driver.o Multiply.o
	$(CXX) Driver.o Multiply.o -o vec_samples_pointer_disam.out $(LIBS)
	@echo "Build complete."
	@echo "You may run ./vec_samples_pointer_disam.out now."

align: CXXFLAGS = -c -O2 -std=c99 -qopt-report=4 -qopt-report-phase=vec -D NOALIAS -D ALIGNED $(INCLUDES) $(BITS)
align: clean Driver.o Multiply.o
	$(CXX) Driver.o Multiply.o -o vec_samples_align.out $(LIBS)
	@echo "Build complete."
	@echo "You may run ./vec_samples_align.out now."

ipo: CXXFLAGS = -c -O2 -std=c99 -ipo -qopt-report=2 -qopt-report-phase=vec -D NOALIAS -D ALIGNED $(INCLUDES) $(BITS)
ipo: clean Driver.o Multiply.o
	$(CXX) -ipo Driver.o Multiply.o -o vec_samples_ipo.out $(LIBS)
	@echo "Build complete."
	@echo "You may run ./vec_samples_ipo.out now."

Driver.o: src/Driver.c
	$(CXX) $(BITS) $(CXXFLAGS) -o $@ src/Driver.c

Multiply.o: src/Multiply.c
	$(CXX) $(BITS) $(CXXFLAGS) -o $@ src/Multiply.c

clean:
	rm -f $(PROGS) *.o *.out *.optrpt
