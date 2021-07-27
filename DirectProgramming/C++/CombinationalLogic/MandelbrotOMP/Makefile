#==============================================================
#
# SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
# http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
#
# Copyright Intel Corporation
#
# THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
#
# =============================================================
CXX := icpc
SRCDIR := src
BUILDDIR := release
CFLAGS := -O3 -ipo -qopenmp -std=c++11
EXTRA_CFLAGS :=
LIBFLAGS := -qopenmp

ifdef perf_num
	EXTRA_CFLAGS += -D PERF_NUM
endif

TARGET := $(BUILDDIR)/Mandelbrot

icpc: $(TARGET)

SOURCES := $(wildcard $(SRCDIR)/*.cpp)
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.cpp=.o))

$(TARGET): $(OBJECTS)
	@echo " Linking..."
	$(CXX) $^ $(LIBFLAGS) -o $(TARGET)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(BUILDDIR)
	$(CXX) -c $(CFLAGS) $(EXTRA_CFLAGS) -o $@ $< 

run: $(TARGET)
ifeq ($(shell uname -s),Darwin)
	@export  DYLD_LIBRARY_PATH="$(LIBRARY_PATH)"; ./$(TARGET) $(option)
else
	./$(TARGET) $(option)
endif

clean:
	@echo " Cleaning..."
	@rm -fr $(BUILDDIR) $(TARGET) 2>/dev/null || true
	@rm -f *.png

.PHONY: clean
