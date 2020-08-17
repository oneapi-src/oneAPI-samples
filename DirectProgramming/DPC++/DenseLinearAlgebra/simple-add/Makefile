CXX = dpcpp
CXXFLAGS = -O2 -g -std=c++17

USM_EXE_NAME = simple-add-usm
USM_SOURCES = src/simple-add-usm.cpp

BUFFER_EXE_NAME = simple-add-buffers
BUFFER_SOURCES = src/simple-add-buffers.cpp

all: build_usm
	
build_usm:
	$(CXX) $(CXXFLAGS) -o $(USM_EXE_NAME) $(USM_SOURCES)

build_buffers:
	$(CXX) $(CXXFLAGS) -o $(BUFFER_EXE_NAME) $(BUFFER_SOURCES)

run:
	./$(USM_EXE_NAME)

run_buffers:
	./$(BUFFER_EXE_NAME)

clean:
	rm -rf $(USM_EXE_NAME) $(BUFFER_EXE_NAME)