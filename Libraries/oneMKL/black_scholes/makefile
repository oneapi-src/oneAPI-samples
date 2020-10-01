# ==============================================================================
# Makefile for NMAKE

# ==============================================================================
#  Content:
#      Black-Scholes formula example makefile
# ==============================================================================
#
#    Parameters of this makefile:
#    ----------------------------  
#      N              : number of stock options
#      ACC=ha, la, ep : VML accuracy level
# ==============================================================================

N = 10000000
ACC = ha

CFLAGS = -I"$(MKLROOT)\include" -Qmkl -EHsc -O2 -fsycl-device-code-split=per_kernel -DACC_$(ACC) -fno-sycl-early-optimizations
LIBS = OpenCL.lib

all: black_scholes.run

black_scholes.exe: black_scholes.cpp
	dpcpp black_scholes.cpp $(CFLAGS) $(LIBS) -o black_scholes.exe

black_scholes.run: black_scholes.exe
	.\black_scholes.exe $(N)

clean:
	del /q black_scholes.exe black_scholes.exp black_scholes.lib
