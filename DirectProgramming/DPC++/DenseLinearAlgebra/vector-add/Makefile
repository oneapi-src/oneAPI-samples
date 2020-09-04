CXX = dpcpp
CXXFLAGS = -O2 -g -std=c++17

BUFFER_EXE_NAME = vector-add-buffers
BUFFER_SOURCES = src/vector-add-buffers.cpp

USM_EXE_NAME = vector-add-usm
USM_SOURCES = src/vector-add-usm.cpp

all: build_buffers

build_buffers:
	$(CXX) $(CXXFLAGS) -o $(BUFFER_EXE_NAME) $(BUFFER_SOURCES)

build_usm:
	$(CXX) $(CXXFLAGS) -o $(USM_EXE_NAME) $(USM_SOURCES)

run: 
	./$(BUFFER_EXE_NAME)

run_usm: 
	./$(USM_EXE_NAME)

clean: 
	rm -f $(BUFFER_EXE_NAME) $(USM_EXE_NAME)
