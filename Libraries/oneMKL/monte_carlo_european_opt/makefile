all: montecarlo

DPCPP_OPTS=/I$(MKLROOT)\include /DMKL_ILP64 -fsycl -fsycl-device-code-split=per_kernel -qmkl

montecarlo: src/montecarlo_main.cpp
	icpx src/montecarlo_main.cpp /omontecarlo.exe $(DPCPP_OPTS)

clean:
	del /q montecarlo.exe

pseudo: clean all
