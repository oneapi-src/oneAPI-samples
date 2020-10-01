CXX = dpcpp
CXXFLAGS = -o
LDFLAGS = -lOpenCL -lsycl
EXE_NAME = simple
SOURCES = src/simple.cpp
BINDIR = bin

all: main

main:
	$(CXX) $(CXXFLAGS) $(BINDIR)/$(EXE_NAME) $(SOURCES) $(LDFLAGS)

run: 
	$(BINDIR)/$(EXE_NAME)

clean: 
	rm -rf $(BINDIR)/$(EXE_NAME)
