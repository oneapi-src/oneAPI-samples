## =============================================================
## Copyright © 2020 Intel Corporation
##
## SPDX-License-Identifier: MIT
## =============================================================

CXX=icx
INCLUDES=-I src
CXXFLAGS=-c -O2 $(INCLUDES) $(BITS)

PROGS=MatVector

help:
	@echo "Usage:"
	@echo "    To build: make <config>"
	@echo "    To run  : MatVector                (for baseline)"
	@echo "              vec_samples_<config>.out (for other configs)"
	@echo ""
	@echo "where <config> is one of the following:"
	@echo "    baseline, vec_report1, vec_report2, vec_report3"
	@echo ""

run: help

baseline: clean driver.o multiply.o
	$(CXX) driver.o multiply.o -o MatVector $(LIBS)
	@echo "Build complete."
	@echo "You may run ./MatVector now."

vec_report1: CXXFLAGS = -c -O2 -std=c17 -qopt-report=1 -DNOFUNCCALL $(INCLUDES)
vec_report1: clean driver.o multiply.o
	$(CXX) driver.o multiply.o -o vec_samples_report1.out $(LIBS)
	@echo "Build complete."
	@echo "You may run ./vec_samples_report1.out now."

vec_report2: CXXFLAGS = -c -O2 -std=c17 -qopt-report=2 -DNOFUNCCALL $(INCLUDES)
vec_report2: clean driver.o multiply.o
	$(CXX) driver.o multiply.o -o vec_samples_report2.out $(LIBS)
	@echo "Build complete."
	@echo "You may run ./vec_samples_report2.out now."

vec_report3: CXXFLAGS = -c -O2 -std=c17 -qopt-report=3 -DNOFUNCCALL $(INCLUDES)
vec_report3: clean driver.o multiply.o
	$(CXX) driver.o multiply.o -o vec_samples_report3.out $(LIBS)
	@echo "Build complete."
	@echo "You may run ./vec_samples_report3.out now."

driver.o: src/driver.c
	$(CXX) $(BITS) $(CXXFLAGS) -o $@ src/driver.c

multiply.o: src/multiply.c
	$(CXX) $(BITS) $(CXXFLAGS) -o $@ src/multiply.c

clean:
	rm -f $(PROGS) *.o *.out *.optrpt
