CC = icx
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
	$(CC) -O2 -msse3 -c -o $@  $<

%_dbg.o: src/%.cpp
	$(CC) -O0 -msse3 -g -c -o $@  $<

clean:
	/bin/rm -f core.* *.o *.exe
