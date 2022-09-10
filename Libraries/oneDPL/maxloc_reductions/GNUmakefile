# Makefile for GNU Make

default: run_all

all: run_all

run_all: maxloc_operator maxloc_implicit maxloc_buffered maxloc_usm
	./maxloc_operator
	./maxloc_implicit
	./maxloc_buffered
	./maxloc_usm

maxloc_operator: maxloc_operator.cpp
	dpcpp $< -o $@

maxloc_implicit: maxloc_implicit.cpp
	dpcpp $< -o $@

maxloc_buffered: maxloc_buffered.cpp
	dpcpp $< -o $@

maxloc_usm: maxloc_usm.cpp
	dpcpp $< -o $@

clean:
	-rm -f maxloc_operator maxloc_implicit maxloc_buffered maxloc_usm

.PHONY: run_all clean all
