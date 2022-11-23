all: montecarlo.exe

DPCPP_OPTS=/I"$(MKLROOT)\include" /Qmkl /DMKL_ILP64 /EHsc -fsycl -fsycl-device-code-split=per_kernel OpenCL.lib

montecarlo: src/montecarlo_main.cpp
	icpx src/montecarlo_main.cpp /Femontecarlo.exe $(DPCPP_OPTS)

clean:
	del /q montecarlo.exe

pseudo: clean all
