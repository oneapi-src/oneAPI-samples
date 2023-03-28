all: montecarlo

!if "$(generator)" == "mrg"
	GENERATOR=/DUSE_MRG
!endif

!if "$(generator)" == "philox"
	GENERATOR=/DUSE_PHILOX
!endif

DPCPP_OPTS=/I$(MKLROOT)\include /DMKL_ILP64 $(GENERATOR) -fsycl -fsycl-device-code-split=per_kernel -qmkl

montecarlo: src/montecarlo_main.cpp
	icpx src/montecarlo_main.cpp /omontecarlo.exe $(DPCPP_OPTS)

clean:
	del /q montecarlo.exe

pseudo: clean all