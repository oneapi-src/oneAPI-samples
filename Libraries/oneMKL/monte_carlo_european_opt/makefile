all: montecarlo

!if "$(generator)" == "mrg"
	GENERATOR=/DUSE_MRG
!endif

!if "$(generator)" == "philox"
	GENERATOR=/DUSE_PHILOX
!endif

!if "$(init_on_host)" == "1"
        INIT_ON_HOST=/DINIT_ON_HOST=1
!endif

DPCPP_OPTS=/I"$(MKLROOT)\include" /DMKL_ILP64 $(GENERATOR) -fsycl $(INIT_ON_HOST) -qmkl
montecarlo_main.cpp
	icpx src/montecarlo_main.cpp /omontecarlo.exe $(DPCPP_OPTS)

clean:
	del /q montecarlo.exe

pseudo: clean all
