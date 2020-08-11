#==============================================================
#
# SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
# http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
#
# Copyright Intel Corporation
#
# THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
#
# =============================================================
CC = icc
EXECS=intrin_dot_sample.exe intrin_double_sample.exe intrin_ftz_sample.exe
DBG_EXECS=intrin_dot_sample_dbg.exe intrin_double_sample_dbg.exe intrin_ftz_sample_dbg.exe

release: $(EXECS)

debug: $(DBG_EXECS)

run: release
	@for i in $(EXECS); do ./$$i; done

debug_run: debug
	@for i in $(DBG_EXECS); do ./$$i; done

intrin_dot_sample.exe: intrin_dot_sample.o
	$(CC) -O2 $^ -o $@

intrin_double_sample.exe: intrin_double_sample.o
	$(CC) -O2 $^ -o $@

intrin_ftz_sample.exe: intrin_ftz_sample.o
	$(CC) -O2 $^ -o $@

intrin_dot_sample_dbg.exe: intrin_dot_sample_dbg.o
	$(CC) -O0 -g $^ -o $@

intrin_double_sample_dbg.exe: intrin_double_sample_dbg.o
	$(CC) -O0 -g $^ -o $@

intrin_ftz_sample_dbg.exe: intrin_ftz_sample_dbg.o
	$(CC) -O0 -g $^ -o $@

%.o: src/%.cpp
	$(CC) -O2 -c -o $@  $<

%_dbg.o: src/%.cpp
	$(CC) -O0 -g -c -o $@  $<

clean:
	/bin/rm -f core.* *.o *.exe
