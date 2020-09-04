CXX = dpcpp
name=complex_mult
CXXFLAGS = -g -o
LDFLAGS = -lOpenCL -lsycl -g
EXE_NAME = complex_mult
SOURCES = lab/${name}.cpp
INCLUDES = src/
BINDIR = bin

all: main

main:
	$(CXX) $(SOURCES) $(CXXFLAGS) $(BINDIR)/$(EXE_NAME) -I$(INCLUDES) $(LDFLAGS)

run:
	$(BINDIR)/$(EXE_NAME)

clean:
	rm -rf $(BINDIR)/$(EXE_NAME) 
 
