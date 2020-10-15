CXX = nvcc
TARGET = needleman_wunsch_cu
SRCS = src/needle.cu
DEPS = src/needle_kernel.cu src/needle.h

# Use predefined implicit rules and add one for *.cu files.
%.o: %.cu
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $< -o $@

$(TARGET): $(SRCS) $(DEPS)
	$(CXX) $(SRCS) -o $@

run: $(TARGET)
	./$(TARGET) 4096 16

.PHONY: clean
clean:
	rm -f $(TARGET) *.o result.txt
