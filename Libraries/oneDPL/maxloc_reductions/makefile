# Makefile for NMAKE

default: run_all

all: run_all

run_all: maxloc_operator.exe maxloc_implicit.exe maxloc_buffered.exe maxloc_usm.exe
	.\maxloc_operator
	.\maxloc_implicit
	.\maxloc_buffered
	.\maxloc_usm

maxloc_operator.exe: maxloc_operator.cpp
	icx-cl -fsycl maxloc_operator.cpp /Femaxloc_operator.exe

maxloc_implicit.exe: maxloc_implicit.cpp
	icx-cl -fsycl maxloc_implicit.cpp /EHsc /Femaxloc_implicit.exe

maxloc_buffered.exe: maxloc_buffered.cpp
	icx-cl -fsycl maxloc_buffered.cpp /EHsc /Femaxloc_buffered.exe

maxloc_usm.exe: maxloc_usm.cpp
	icx-cl -fsycl maxloc_usm.cpp /EHsc /Femaxloc_usm.exe

clean:
	del /q maxloc_operator.exe maxloc_implicit.exe maxloc_buffered.exe maxloc_usm.exe
	del /q maxloc_operator.exp maxloc_implicit.exp maxloc_buffered.exp maxloc_usm.exp

pseudo: run_all clean all

