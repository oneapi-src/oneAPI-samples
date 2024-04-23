# Makefile for GNU Make

default: run_all

all: run_all

run_all: maxloc_operator maxloc_implicit maxloc_buffered maxloc_usm
	./maxloc_operator
	./maxloc_implicit
	./maxloc_buffered
	./maxloc_usm

maxloc_operator: maxloc_operator.cpp
	icpx $< -fsycl -o $@

maxloc_implicit: maxloc_implicit.cpp
	icpx $< -fsycl -o $@

maxloc_buffered: maxloc_buffered.cpp
	icpx $< -fsycl -o $@

maxloc_usm: maxloc_usm.cpp
	icpx $< -fsycl -o $@

clean:
	-rm -f maxloc_operator maxloc_implicit maxloc_buffered maxloc_usm

.PHONY: run_all clean all
