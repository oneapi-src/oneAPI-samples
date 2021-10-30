CPP_FLAGS:= --gcc-toolchain=${GCC_ROOTDIR} -fintelfpga -fsycl-device-code-split=none

.PHONY: default

default: hw emulator

lib_rtl.o: lib_rtl_spec.xml lib_rtl_model.cpp
	fpga_crossgen lib_rtl_spec.xml --emulation_model lib_rtl_model.cpp --target sycl -o lib_rtl.o

lib.a: lib_rtl.o
	fpga_libtool lib_rtl.o --target sycl --create lib.a

emulator: lib.a use_library.cpp lib_sycl.cpp
	dpcpp ${CPP_FLAGS} -DFPGA_EMULATOR use_library.cpp lib_sycl.cpp lib.a -o $@

hw: lib.a use_library.cpp lib_sycl.cpp
	dpcpp ${CPP_FLAGS} -Xshardware -fsycl-link=early use_library.cpp lib_sycl.cpp lib.a -o $@ -Xssave-temps

clean:
	rm *.o *.a emulator *.prj hw -rf
