CXX = icpx
CXXFLAGS = -std=c++17 -g  -fsycl -o 
LDFLAGS =
EXE_NAME = complex_mult
SOURCES = src/complex_mult.cpp

all: main

main:
	$(CXX) $(CXXFLAGS) $(EXE_NAME) $(SOURCES) $(LDFLAGS)

run: 
	./$(EXE_NAME)

clean: 
	rm -rf $(EXE_NAME)
