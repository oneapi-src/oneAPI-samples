TARGET   = makefile-gpu
SRCS     = src/main.cpp

CXX      = dpcpp
CXXFLAGS = -I. -std=c++17
LDFLAGS  =

ifeq ($(BUILD_MODE), release)
    CXXFLAGS += -O2
else
    CXXFLAGS += -g
endif

.PHONY: all build run clean
.DEFAULT_GOAL := all

# the same as build target
all: build

# build the project
build: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

# run binary
run: $(TARGET)
	./$(TARGET)

# clean all
clean:
	-$(RM) $(TARGET)